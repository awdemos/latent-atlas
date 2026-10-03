//! Model access. The trait takes the full Example so mock clients can use
//! gold fields; real clients only read `example.prompt`.

use crate::types::Example;
use async_trait::async_trait;

#[derive(Clone, Debug, PartialEq)]
pub struct YesNoScore {
    pub logit_diff: f64,
    pub p_yes: f64,
    pub top_logprobs: Vec<(String, f64)>,
}

#[async_trait]
pub trait ModelClient: Send + Sync {
    async fn score(&self, example: &Example) -> anyhow::Result<YesNoScore>;
    fn name(&self) -> &str;
}

fn mass_for(top: &[(String, f64)], word: &str) -> f64 {
    top.iter()
        .filter(|(tok, _)| tok.trim().eq_ignore_ascii_case(word))
        .map(|(_, lp)| lp.exp())
        .sum()
}

/// Aggregate Yes/No probability mass from top-logprobs token variants.
pub fn aggregate_yes_no(top: &[(String, f64)]) -> Option<YesNoScore> {
    let yes = mass_for(top, "yes");
    let no = mass_for(top, "no");
    if yes == 0.0 || no == 0.0 {
        return None;
    }
    Some(YesNoScore {
        logit_diff: yes.ln() - no.ln(),
        p_yes: yes / (yes + no),
        top_logprobs: top.to_vec(),
    })
}

/// Pull top_logprobs[(token, logprob)] out of a chat-completions response.
pub fn parse_chat_response(json: &serde_json::Value) -> anyhow::Result<Vec<(String, f64)>> {
    let tops = json["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("no top_logprobs in response: {json}"))?;
    Ok(tops
        .iter()
        .filter_map(|t| Some((t["token"].as_str()?.to_string(), t["logprob"].as_f64()?)))
        .collect())
}

/// Deterministic mock: p_yes is a logistic function of signed distance to the
/// true interval plus hash-based noise. Powers demos and tests with no network.
pub struct MockClient {
    name: String,
}

impl MockClient {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl ModelClient for MockClient {
    fn name(&self) -> &str {
        &self.name
    }

    async fn score(&self, example: &Example) -> anyhow::Result<YesNoScore> {
        let year = example.year_astronomical;
        let d: i64 = match (example.interval_start, example.interval_end) {
            (Some(a), Some(b)) => {
                if year < a {
                    (year - a) as i64
                } else if year > b {
                    (b - year) as i64
                } else {
                    (year - a).min(b - year) as i64
                }
            }
            (Some(a), None) => (year - a) as i64,
            (None, _) => 0,
        };
        let h = crate::splits::fnv1a64(format!("{}|{}", self.name, example.example_id).as_bytes());
        let noise = (h % 1000) as f64 / 1000.0 - 0.5; // [-0.5, 0.5)
        let z = d as f64 / 8.0 + noise;
        let p = (1.0 / (1.0 + (-z).exp())).clamp(1e-6, 1.0 - 1e-6);
        Ok(YesNoScore {
            logit_diff: (p / (1.0 - p)).ln(),
            p_yes: p,
            top_logprobs: vec![
                ("Yes".to_string(), p.ln()),
                ("No".to_string(), (1.0 - p).ln()),
            ],
        })
    }
}

/// OpenAI-compatible backend: POST {base_url}/chat/completions with
/// max_tokens=1, logprobs=true, top_logprobs=20.
pub struct OpenAiClient {
    base_url: String,
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    pub fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        let base_url = std::env::var("ATLAS_BASE_URL")
            .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let api_key = std::env::var("ATLAS_API_KEY").unwrap_or_default();
        let model = model
            .or_else(|| std::env::var("ATLAS_MODEL").ok())
            .ok_or_else(|| anyhow::anyhow!("model required: pass --model or set ATLAS_MODEL"))?;
        Ok(Self {
            base_url,
            api_key,
            model,
            http: reqwest::Client::new(),
        })
    }
}

#[async_trait]
impl ModelClient for OpenAiClient {
    fn name(&self) -> &str {
        &self.model
    }

    async fn score(&self, example: &Example) -> anyhow::Result<YesNoScore> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": example.prompt}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": true,
            "top_logprobs": 20,
        });
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let json: serde_json::Value = resp.json().await?;
        if !status.is_success() {
            anyhow::bail!("chat/completions HTTP {status}: {json}");
        }
        let top = parse_chat_response(&json)?;
        aggregate_yes_no(&top).ok_or_else(|| yes_no_absent_error(&example.example_id, &top))
    }
}

/// Builds the diagnostic for a probe whose top-logprobs contained neither
/// "Yes" nor "No" — almost always a reasoning/thinking model whose chat
/// template opens with a think-token, making single-token probing impossible
/// at the first position.
fn yes_no_absent_error(example_id: &str, top: &[(String, f64)]) -> anyhow::Error {
    let observed: Vec<&str> = top.iter().map(|(t, _)| t.as_str()).take(8).collect();
    anyhow::anyhow!(
        "Yes/No absent from top_logprobs for {example_id} \
         (observed top tokens: {observed:?}; if the model's first token is \
         never Yes/No it is likely a reasoning/thinking model, which cannot \
         be single-token probed — use a non-thinking model or an endpoint \
         where thinking can be disabled, e.g. vLLM's chat_template_kwargs)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::querygen::{GenConfig, gen_eval};
    use crate::types::tests_helpers::caesar_interval;

    #[test]
    fn aggregate_combines_token_variants() {
        let top = vec![
            (" Yes".to_string(), -0.1f64),
            ("No".to_string(), -3.5),
            (" yes".to_string(), -2.0),
            (" no".to_string(), -2.5),
            ("The".to_string(), -8.0),
        ];
        let s = aggregate_yes_no(&top).unwrap();
        let yes_mass = (-0.1f64).exp() + (-2.0f64).exp();
        let no_mass = (-3.5f64).exp() + (-2.5f64).exp();
        let want = yes_mass / (yes_mass + no_mass);
        assert!((s.p_yes - want).abs() < 1e-9);
        assert!((s.logit_diff - (yes_mass.ln() - no_mass.ln())).abs() < 1e-9);
        assert!(s.p_yes > 0.9);
    }

    #[test]
    fn aggregate_returns_none_when_a_side_is_missing() {
        let top = vec![("Yes".to_string(), -0.01f64), ("The".to_string(), -5.0)];
        assert!(aggregate_yes_no(&top).is_none());
    }

    #[test]
    fn parse_chat_response_extracts_top_logprobs() {
        let json = serde_json::json!({"choices": [{"logprobs": {"content": [
            {"token": "Yes", "logprob": -0.05, "top_logprobs": [
                {"token": "Yes", "logprob": -0.05},
                {"token": "No", "logprob": -3.2}
            ]}
        ]}}]});
        let top = parse_chat_response(&json).unwrap();
        assert_eq!(
            top,
            vec![("Yes".to_string(), -0.05), ("No".to_string(), -3.2)]
        );
    }

    #[tokio::test]
    async fn mock_scores_inside_high_outside_low() {
        let client = MockClient::new("mock-v1");
        let iv = caesar_interval();
        let examples = gen_eval(&iv, "Julius Caesar", &[], &GenConfig::default());
        for ex in &examples {
            let s = client.score(ex).await.unwrap();
            // strict checks only where the mock is unambiguous: near bands sit
            // inside the mock's transition slope and may legitimately straddle 0.5
            match ex.sample_band.as_str() {
                "interior" => assert!(s.p_yes > 0.5, "{ex:?}"),
                "far_before" | "far_after" | "far_fallback" => assert!(s.p_yes < 0.5, "{ex:?}"),
                _ => assert!(s.p_yes > 0.0 && s.p_yes < 1.0),
            }
            assert!((s.logit_diff - (s.p_yes / (1.0 - s.p_yes)).ln()).abs() < 1e-9);
        }
    }

    #[tokio::test]
    async fn mock_is_deterministic() {
        let client = MockClient::new("mock-v1");
        let iv = caesar_interval();
        let examples = gen_eval(&iv, "Julius Caesar", &[], &GenConfig::default());
        let a = client.score(&examples[0]).await.unwrap();
        let b = client.score(&examples[0]).await.unwrap();
        assert_eq!(a.p_yes, b.p_yes);
    }

    #[test]
    fn yes_no_absent_error_lists_observed_tokens() {
        let top = vec![("Thinking".to_string(), 0.0), ("Okay".to_string(), -6.5)];
        let err = yes_no_absent_error("example_42", &top).to_string();
        assert!(err.contains("example_42"), "{err}");
        assert!(err.contains("Thinking"), "{err}");
        assert!(err.contains("thinking model"), "{err}");
    }
}
