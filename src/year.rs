//! Astronomical year numbering: 1 BCE = 0, 2 BCE = -1, 100 BCE = -99.
//! Internal i32 is always astronomical; prompts use BCE/CE display strings.

pub fn display_year(y: i32) -> String {
    if y <= 0 {
        format!("{} BCE", 1 - y)
    } else {
        format!("{y} CE")
    }
}

pub fn parse_display_year(s: &str) -> Option<i32> {
    let s = s.trim();
    if let Some(rest) = s.strip_suffix("BCE") {
        let n: i32 = rest.trim().parse().ok()?;
        (n >= 1).then_some(1 - n)
    } else if let Some(rest) = s.strip_suffix("CE") {
        let n: i32 = rest.trim().parse().ok()?;
        (n >= 1).then_some(n)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_handles_bce_ce_and_year_zero() {
        assert_eq!(display_year(-99), "100 BCE");
        assert_eq!(display_year(-1), "2 BCE");
        assert_eq!(display_year(0), "1 BCE");
        assert_eq!(display_year(1), "1 CE");
        assert_eq!(display_year(44), "44 CE");
        assert_eq!(display_year(2026), "2026 CE");
    }

    #[test]
    fn parse_roundtrips() {
        for y in [-99, -1, 0, 1, 44, 1453, 2026] {
            assert_eq!(parse_display_year(&display_year(y)), Some(y));
        }
    }

    #[test]
    fn parse_rejects_garbage() {
        assert_eq!(parse_display_year("year 100"), None);
        assert_eq!(parse_display_year("0 BCE"), None);
        assert_eq!(parse_display_year("3 CE 4"), None);
        assert_eq!(parse_display_year(""), None);
    }
}
