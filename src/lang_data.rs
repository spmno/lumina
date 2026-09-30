//! Static tables for language/script detection, transcribed from laya's `laya/lang.py`
//! (v0.3.21). Order is significant where noted: language tie-breaks follow the stopword
//! list order, and script counts append `latin` last.

// The items below are crate-only through module privacy; `pub(crate)` is spelled out
// because `unreachable_pub` is deliberate `[lints.rust]` config and its suggestion is
// exactly this — the nursery `redundant_pub_crate` disagrees and is silenced here.
#![allow(clippy::redundant_pub_crate)]

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Unicode blocks the English checkpoint cannot read: `(script, ranges)`.
pub(crate) static SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function words per language. The Romance lists deliberately carry the *unaccented*
/// forms as well as the accented ones: a state that lost its accents (mail clients,
/// ASCII-normalising pipelines) keeps only them as evidence. Order is laya's `_STOP`
/// literal order and is load-bearing: language ties resolve to the earlier language.
const STOP_RAW: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
            "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton", "ma",
            "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien", "tout",
            "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit", "merci",
            "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi", "alors",
            "donc",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
            "ich", "wir", "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen", "meinem",
            "meiner", "diese", "dieser", "diesen", "dieses", "einen", "einem", "einer", "wie",
            "wo", "wann", "welche", "im", "zum", "zur", "aus", "bei", "nach", "noch", "bitte",
            "heute", "jetzt", "kann", "kannst", "habe", "gibt", "wird", "in", "was",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus", "la", "un",
            "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos", "ni", "dos", "tres", "fue",
            "fueron", "ser", "tiene", "tienen", "tengo", "puede", "pueden", "quiero", "necesito",
            "hemos", "han", "sobre", "entre", "cuando", "donde", "porque", "aunque", "también",
            "ya", "eso", "esto", "esa", "ese", "nada", "algo", "aquí", "hoy", "gracias",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela", "o", "e",
            "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou", "tem", "tenho",
            "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu", "sua", "isso",
            "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais", "já", "ainda",
            "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado", "olá", "você",
            "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
            "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
            "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite", "depois", "antes",
            "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum", "nenhuma",
            "estava", "ficou", "fiz", "deu",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla", "la", "le", "un", "uno",
            "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi", "ci", "ne", "ho", "hai", "ha",
            "abbiamo", "avete", "hanno", "era", "stato", "stata", "devo", "deve", "devono",
            "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando", "dove", "perche", "molto",
            "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi", "ieri", "grazie", "ciao",
            "scusa", "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo",
            "dagli", "dei", "delle", "dello", "degli", "agli", "alle", "col",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
    // Romanian words its Romance neighbours do not share; `la`, `o`, `un`, `de`, `pe`,
    // `ca` are deliberately absent so adding `ro` cannot steal a French/Spanish state.
    (
        "ro",
        &[
            "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
            "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
            "acesta", "aceasta", "mi", "ți", "vă", "nu",
        ],
    ),
    // Romanized Bangla ("Banglish"): how Bangla is typed with no Bengali keyboard.
    // Frequent Bangla words that are also English words are deliberately absent.
    (
        "bn",
        &[
            "ami",
            "amar",
            "amake",
            "amra",
            "amader",
            "apni",
            "apnar",
            "apnake",
            "apnara",
            "tumi",
            "tomar",
            "tomake",
            "tomra",
            "tader",
            "ota",
            "eita",
            "oita",
            "ekta",
            "ei",
            "oi",
            "ki",
            "keno",
            "kivabe",
            "kibhabe",
            "kothay",
            "kokhon",
            "kobe",
            "koto",
            "kintu",
            "jodi",
            "tahole",
            "ar",
            "theke",
            "jonno",
            "sathe",
            "shathe",
            "diye",
            "niye",
            "moddhe",
            "kore",
            "korte",
            "korchi",
            "korsi",
            "korbo",
            "korechi",
            "koreche",
            "korun",
            "koren",
            "korlam",
            "hobe",
            "hoyeche",
            "hoise",
            "hocche",
            "hoyni",
            "chai",
            "chaina",
            "lagbe",
            "parchi",
            "parbo",
            "parchina",
            "peyechi",
            "paini",
            "dite",
            "dilam",
            "diyechi",
            "nai",
            "khub",
            "onek",
            "ekhon",
            "akhon",
            "ekhono",
            "abar",
            "ekbar",
            "duibar",
            "ajke",
            "kalke",
            "taka",
            "bhalo",
            "valo",
            "kharap",
            "shomossa",
            "somossa",
            "dhonnobad",
            "bhai",
            "shob",
            "keu",
            "kichu",
            "bolte",
            "bolun",
            "parben",
            "asbe",
            "jabe",
            "pabo",
            "ferot",
            "dorkar",
            "hoye",
            "geche",
            "gese",
        ],
    ),
    (
        "az",
        &[
            "və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa", "var",
            "yox", "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox", "hər",
            "nə", "kimi", "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma", "ancaq",
            "artıq", "artiq", "də", "isə", "həm", "yalnız", "yalniz",
        ],
    ),
];

/// Letters that ordinary English does not use: the signal that catches a Latin-script
/// language held in no stopword list (Romanian, Polish, Czech, Turkish, Baltic, ...).
/// Grouped by language family as in laya's source.
pub(crate) const NON_EN_DIACRITICS: &str = concat!(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ", // Western European
    "ăâîșțşţ",                         // Romanian
    "ąćęłńśźż",                        // Polish
    "čďěňřšťůž",                       // Czech / Slovak
    "őű",                              // Hungarian
    "ğı",                              // Turkish (text is lowercased before matching)
    "āēģīķļņūž",                       // Baltic
    "đ",                               // Serbo-Croatian / Vietnamese
    "ə",                               // Azerbaijani
);

pub(crate) fn stop_sets() -> &'static Vec<(&'static str, HashSet<&'static str>)> {
    static STOP: OnceLock<Vec<(&'static str, HashSet<&'static str>)>> = OnceLock::new();
    STOP.get_or_init(|| {
        STOP_RAW
            .iter()
            .map(|(lg, words)| (*lg, words.iter().copied().collect()))
            .collect()
    })
}

/// The stopword set of one language, or `None` for a language laya does not hold.
pub(crate) fn stop_set(lang: &str) -> Option<&'static HashSet<&'static str>> {
    stop_sets()
        .iter()
        .find(|(lg, _)| *lg == lang)
        .map(|(_, s)| s)
}

/// Words more than one stopword list claims: a shared word may not name a winner by
/// itself (Romanian text was once reported as French that way), though it still counts
/// toward the total of a language that also matched a word of its own.
pub(crate) fn shared_words() -> &'static HashSet<&'static str> {
    static SHARED: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SHARED.get_or_init(|| {
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for (_, words) in stop_sets() {
            for w in words {
                *counts.entry(w).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .filter(|(_, n)| *n > 1)
            .map(|(w, _)| w)
            .collect()
    })
}

/// English function words no other list holds. They alone carry the English rescue of
/// `latin_profile`.
pub(crate) fn en_only_words() -> &'static HashSet<&'static str> {
    static EN_ONLY: OnceLock<HashSet<&'static str>> = OnceLock::new();
    EN_ONLY.get_or_init(|| {
        let en: HashSet<&'static str> = stop_sets()
            .iter()
            .find(|(lg, _)| *lg == "en")
            .map_or_else(HashSet::new, |(_, words)| words.iter().copied().collect());
        en.difference(shared_words()).copied().collect()
    })
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]

    use super::*;

    #[test]
    fn stops_are_in_laya_order() {
        let lgs: Vec<&str> = STOP_RAW.iter().map(|(lg, _)| *lg).collect();
        assert_eq!(
            lgs,
            vec!["en", "fr", "de", "es", "pt", "it", "nl", "ro", "bn", "az"]
        );
    }

    #[test]
    fn shared_words_match_python_computation() {
        // "la" is claimed by fr, es, it; "que" by fr, es, pt; "se" by es, pt.
        let shared = shared_words();
        for w in ["la", "que", "se", "una", "mais", "como"] {
            assert!(shared.contains(w), "expected {w} shared");
        }
        for w in ["the", "evidenced", "mein", "você"] {
            assert!(!shared.contains(w), "expected {w} unshared");
        }
    }

    #[test]
    fn en_only_words_exclude_shared() {
        // "in"/"was" are shared with de, "is" with nl; "please" is English alone.
        let en_only = en_only_words();
        for w in ["in", "is", "was"] {
            assert!(!en_only.contains(w), "expected {w} shared");
        }
        for w in ["please", "which", "have", "not"] {
            assert!(en_only.contains(w), "expected {w} English-only");
        }
    }

    #[test]
    fn diacritics_contain_laya_letters() {
        for c in ['é', 'ü', 'ß', 'ș', 'ż', 'ğ', 'ə', 'đ', 'í'] {
            assert!(NON_EN_DIACRITICS.contains(c), "expected {c}");
        }
        for c in ['a', 'e', 'z', 'x'] {
            assert!(!NON_EN_DIACRITICS.contains(c), "unexpected {c}");
        }
    }

    #[test]
    fn script_ranges_cover_laya_blocks() {
        let find = |cp: u32| {
            SCRIPT_RANGES
                .iter()
                .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi))
                .map(|(name, _)| *name)
        };
        assert_eq!(find(0x0915), Some("devanagari"));
        assert_eq!(find(0x0410), Some("cyrillic"));
        assert_eq!(find(0xAC00), Some("hangul"));
        assert_eq!(find(0x0041), None); // Latin 'A'
        assert_eq!(find(0x30C3), Some("kana")); // katakana ッ
    }
}
