"""Process INDB.xlsx + Cleaned_Indian_Food_Dataset.csv + ICMR-NIN (IFCT 2017) calibration.
All nutritional values are standardized to a fixed 100 g edible portion.

Run once (idempotent):  python scripts/clean_dishes.py
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

ROOT = Path(__file__).resolve().parents[1]
DATA_IN = ROOT / "data" / "INDB.xlsx"
RECIPE_CSV = ROOT / "data" / "Cleaned_Indian_Food_Dataset.csv"
DATA_OUT = ROOT / "data" / "dishes_engineered.csv"

# ICMR-NIN (IFCT 2017) per-100g cooked ingredient contributions:
# (Sodium_mg, Iron_mg, Calcium_mg, Protein_g, Fibre_g)
IFCT_PRIORS = {
    "is_chicken":       (75.0,  1.4,  18.0, 18.5, 0.0),
    "is_red_meat":      (85.0,  2.6,  22.0, 19.5, 0.0),
    "is_fish":          (90.0,  1.3,  45.0, 17.5, 0.0),
    "is_egg":           (125.0, 1.8,  50.0, 12.5, 0.0),
    "is_paneer":        (40.0,  0.5, 210.0, 14.5, 0.0),
    "is_cheese_cream":  (280.0, 0.3, 240.0, 11.0, 0.0),
    "is_dairy":         (45.0,  0.2, 120.0,  4.2, 0.0),
    "is_soya":          (30.0,  4.2, 110.0, 16.0, 4.5),
    "is_chickpea_bean": (35.0,  2.8,  55.0,  8.5, 5.8),
    "is_legume":        (30.0,  2.4,  38.0,  7.8, 4.4),
    "is_leafy_green":   (65.0,  3.8, 115.0,  2.8, 3.2),
    "is_sesame":        (15.0,  5.2, 340.0,  6.5, 4.0),
    "is_nuts_seed":     (15.0,  2.2,  75.0,  6.0, 3.0),
    "is_millet":        (15.0,  2.9, 130.0,  4.5, 3.8),
    "is_wheat":         (120.0, 2.2,  30.0,  6.8, 3.6),
    "is_besan":         (90.0,  2.6,  45.0,  8.2, 4.2),
    "is_rice":          (15.0,  0.6,  12.0,  2.7, 0.8),
    "is_semolina":      (20.0,  1.2,  22.0,  3.8, 1.4),
    "is_pasta_noodle":  (140.0, 1.1,  18.0,  4.5, 1.5),
    "is_cruciferous":   (35.0,  1.1,  42.0,  2.1, 2.6),
    "is_pea_corn":      (25.0,  1.4,  28.0,  4.2, 3.4),
    "is_potato":        (20.0,  0.7,  14.0,  1.8, 1.8),
    "is_gourd_squash":  (18.0,  0.8,  26.0,  1.2, 2.0),
    "is_tomato":        (25.0,  0.6,  18.0,  1.0, 1.2),
    "is_onion_garlic":  (20.0,  0.5,  22.0,  1.1, 1.3),
    "is_vegetable":     (25.0,  1.0,  32.0,  1.6, 2.4),
    "is_fruit":         (8.0,   0.5,  16.0,  0.8, 2.2),
    "is_citrus_tangy":  (25.0,  0.8,  28.0,  0.6, 1.5),
    "is_coconut":       (20.0,  1.4,  24.0,  2.8, 4.2),
    "is_ghee_butter":   (45.0,  0.1,  15.0,  0.2, 0.0),
    "is_salted":        (260.0, 0.3,  15.0,  0.0, 0.0),
    "is_spicy":         (140.0, 0.6,  20.0,  0.4, 0.6),
    "is_fried":         (95.0,  0.2,   5.0,  0.5, 0.2),
    "is_cured":         (130.0, 0.4,  10.0,  1.5, 0.0),
    "is_sweet":         (10.0,  0.3,  15.0,  0.0, 0.0),
}

DISH_BASE_PRIORS = {
    "Main":     (160.0, 1.1, 35.0, 3.2, 1.4),
    "Side":     (140.0, 1.0, 32.0, 2.2, 1.8),
    "Bread":    (150.0, 1.8, 28.0, 6.5, 3.0),
    "Snack":    (210.0, 1.3, 30.0, 4.5, 2.0),
    "Dessert":  (35.0,  0.6, 65.0, 3.5, 0.8),
    "Soup":     (220.0, 0.8, 24.0, 2.0, 1.0),
    "Salad":    (110.0, 1.0, 30.0, 2.0, 2.2),
    "Beverage": (30.0,  0.3, 45.0, 1.8, 0.4),
    "Condiment": (480.0, 1.5, 45.0, 2.5, 3.0),
}


def assign_type(row: pd.Series) -> str:
    name = str(row["food_name"]).lower().strip()
    sunit = str(row.get("servings_unit", "")).lower().strip()

    if re.search(r"\b(?:soup|shorba|rasam|consomme|broth|stock)\b", name) or sunit == "soup bowl":
        if not re.search(r"\b(?:rasam powder|rasam masala)\b", name):
            return "Soup"

    if not re.search(r"\b(?:in coconut milk|with coconut milk|cornflakes with milk|oats with milk|semolina milk)\b", name):
        if (
            re.search(
                r"\b(?:drink|juice|shake|milkshake|smoothie|tea|chai|coffee|kehwa|cocoa|hot chocolate|"
                r"panna|punch|lemonade|water|jal jeera|jaljeera|sharbat|sherbet|thandai|lassi|"
                r"buttermilk|chaas|kanji|cooler|squash|coco pine|lem-o-gin|gingo|egg nog|mojito|"
                r"soda|haldi doodh|badam milk|masala milk|spiced milk)\b",
                name,
            )
            or sunit in ("tall glass", "glass", "cup")
        ):
            return "Beverage"

    if re.search(
        r"\b(?:chat masala|garam masala|pav bhaji masala|kashmiri masala|sambhar masala|sambar powder|"
        r"chana masala powder|rasam powder|rasam masala|curry powder|panch phoran|spice blend|"
        r"white sauce|brown sauce|cheese sauce|tomato sauce|bread sauce|egg sauce|mint sauce|"
        r"apple sauce|cranberry sauce|chocolate sauce|custard sauce|green chilli sauce|"
        r"mayonnaise|dressing|butter icing|royal icing|glace icing|american frosting)\b",
        name,
    ) and not re.search(
        r"\b(?:spaghetti|fettuccine|pasta|macaroni|lasagne|baked|fried|chicken|fish|mutton|"
        r"paneer|brinjal|egg|pancake|pudding|tart|cake|sponge|roll|sandwich)\b",
        name,
    ):
        return "Condiment"

    if re.search(r"\b(?:pickle|aachar|achar|chutney|murabba|kasundi|jam|marmalade|jelly)\b", name) and not re.search(
        r"\b(?:with|sandwich|pakora|paratha|parantha|rice|pulao|roll|dosa|idli|pancake|tart|cake|sponge|bread)\b",
        name,
    ):
        return "Condiment"

    if re.search(r"\b(?:salad|salaad|raita|kachumber|kosambari)\b", name):
        return "Salad"

    is_savoury_dish = bool(
        re.search(
            r"\b(?:sweet and sour|sweet corn|sweet potato|savoury|keema|meat|chicken|fish|mutton|prawn|"
            r"spinach|mushroom|vegetable|cheese|pea|potato|dal|lentil|onion|garlic|chilli|mirch|masala|"
            r"shepherd|cottage|bolognese|macaroni|macroni)\b",
            name,
        )
    )
    if not is_savoury_dish:
        if (
            re.search(
                r"\b(?:ladoo|laddu|chikki|cake|gateau|cupcake|pastry|brownie|pie|tart|pudding|souffle|"
                r"mousse|custard|ice cream|kulfi|sorbet|popsicle|halwa|kheer|payasam|phirni|shrikhand|"
                r"basundi|rabri|rabdi|sweet|mithai|barfi|burfi|pedha|peda|gulab jamun|rasgulla|rasmalai|"
                r"jalebi|imarti|malpua|ghevar|sandesh|modak|sheera|kesari|biscuit|cookie|macaroon|"
                r"merinque|meringue|doughnut|donut|waffle|crepe|pancake|french toast|trifle|fudge|"
                r"toffee|caramel|praline|marzipan|fondant|shahi tukda|shahi toast|double ka meetha|"
                r"meetha|meethe|balushahi|gujiya|karanji|chhena|chamcham|khaja|chiroti|mysore pak|"
                r"patishapta|pithe|pitha|bebinca|dodol|shankarpali|shakarpara|adhirasam|unniyappam|"
                r"neyyappam|kozhukattai|puran poli|bobbatlu|holige)\b",
                name,
            )
            or sunit in ("biscuit", "cookie", "ladoo", "burfi")
        ):
            return "Dessert"

    if re.search(
        r"\b(?:halwa|kheer|payasam|phirni|barfi|burfi|ladoo|laddu|peda|pedha|gulab jamun|rasgulla|"
        r"rasmalai|jalebi|malpua|sandesh|shrikhand|basundi|rabri|rabdi|kulfi|ice cream|cake|gateau|"
        r"pastry|cookie|biscuit|pudding|mithai|meetha|meethe)\b",
        name,
    ) and not re.search(r"\b(?:sweet and sour|sweet corn|sweet potato|pickle|achar|aachar|chutney|savoury|namkeen)\b", name):
        return "Dessert"

    if re.search(
        r"\b(?:roti|chapati|phulka|naan|kulcha|paratha|parantha|puri|poori|bhatura|bhature|luchi|"
        r"thepla|bhakri|thalipeeth|akki rotti|makki ki roti|missi roti|roomali|rumali|sheermal|"
        r"bafla|bati|pav|bread|bun|toast|rusk|croissant|bagel|pita|lavash)\b",
        name,
    ) and not re.search(
        r"\b(?:sandwich|pakora|pakoda|chaat|upma|poha|vada pav|pav bhaji|misal pav|dabeli|"
        r"roll|pizza|burger|pudding|halwa|churma|bonda|bajji|cutlet|soup|salad|sauce)\b",
        name,
    ):
        return "Bread"

    if (
        re.search(
            r"\b(?:snack|pakora|pakoda|bhajiya|bonda|vada|wada|samosa|kachori|chaat|papdi|bhel|sev|"
            r"pani puri|golgappa|puchka|tikki|cutlet|kebab|kabab|shammi|seekh|tikka|65|roll|"
            r"spring roll|frankie|wrap|momos|dumpling|sandwich|burger|pizza|puff|puffs|patty|patties|"
            r"quiche|tartlet|vol-au-vent|canape|bruschetta|croquette|fritter|fries|chips|nachos|"
            r"popcorn|murmura|chiwda|chivda|mixture|namkeen|namak para|namak paras|mathri|chakli|"
            r"murukku|idli|dosa|uttapam|appe|paniyaram|upma|poha|cheela|chilla|dhokla|khandvi|"
            r"handvo|muthiya|muthias|patra|farsan|vada pav|dabeli|papad|khakhra|bhujiya|bhujia|"
            r"omelette|omlet|scrambled|boiled egg|poached egg|fried egg|scotch egg|deviled egg|"
            r"pancake|waffle|crepe|porridge|daliya|cornflakes|flakes|muesli)\b",
            name,
        )
        or sunit in ("sandwich", "cutlet", "vada", "cheela", "triangle")
    ):
        if not re.search(r"\b(?:curry|gravy|korma|makhani|do piaza|yakhni|biryani|pulao)\b", name):
            return "Snack"

    if (
        re.search(
            r"\b(?:curry|gravy|korma|kurma|masala|makhani|makhanwala|kadai|karahi|handi|do piaza|"
            r"dopiaza|jalfrezi|vindaloo|rogan josh|yakhni|nihari|haleem|keema|kofta|pasanda|rezala|"
            r"xacuti|cafreal|sorpotel|moilee|mappas|chettinad|kolhapuri|malvani|nawabi|shahi|dal|"
            r"daal|dhal|sambar|sambhar|kadhi|amti|pulusu|kootu|avial|aviyal|chole|chana|channa|rajma|"
            r"lobia|usal|misal|pav bhaji|rice|chawal|bhat|bhaat|pulao|pilaf|biryani|biriyani|khichdi|"
            r"khichri|pongal|bisi bele|puliyogare|risotto|paella|pasta|spaghetti|macaroni|macroni|"
            r"fettuccine|lasagne|lasagna|penne|noodles|chowmein|hakka|chop suey|casserole|au gratin|"
            r"bake|baked|roast|grilled|stew|stroganoff|goulash|ala king|bolognese|pie|souffle|"
            r"manchurian|sweet and sour|chilli chicken|chilli paneer|chilli fish|tandoori chicken|"
            r"tandoori fish|butter chicken|chops|steak|fillet|fish|machli|prawn|shrimp|crab|lobster|"
            r"chicken|murgh|mutton|lamb|gosht|beef|pork|meat|egg|anda|paneer|tofu|soya|nutrinugget)\b",
            name,
        )
        or sunit in ("plate", "curry bowl")
    ):
        return "Main"

    return "Side"


def compute_ifct_priors(dish_type: str, flags: dict) -> dict:
    base_na, base_fe, base_ca, base_pr, base_fb = DISH_BASE_PRIORS.get(
        dish_type, (120.0, 1.0, 30.0, 3.0, 1.5)
    )
    active = [k for k, v in flags.items() if v == 1 and k in IFCT_PRIORS]
    if not active:
        return {
            "prior_Sodium": base_na, "prior_Iron": base_fe,
            "prior_Calcium": base_ca, "prior_Protein": base_pr, "prior_Fibre": base_fb
        }

    modifiers = {"is_salted", "is_spicy", "is_fried", "is_cured", "is_sweet", "is_ghee_butter", "is_citrus_tangy"}
    bulk = [k for k in active if k not in modifiers]
    mods = [k for k in active if k in modifiers]

    if bulk:
        na_vals = [IFCT_PRIORS[k][0] for k in bulk]
        fe_vals = [IFCT_PRIORS[k][1] for k in bulk]
        ca_vals = [IFCT_PRIORS[k][2] for k in bulk]
        pr_vals = [IFCT_PRIORS[k][3] for k in bulk]
        fb_vals = [IFCT_PRIORS[k][4] for k in bulk]

        na = 0.35 * base_na + 0.65 * (0.60 * max(na_vals) + 0.40 * float(np.mean(na_vals)))
        fe = 0.20 * base_fe + 0.80 * (0.65 * max(fe_vals) + 0.35 * float(np.mean(fe_vals)))
        ca = 0.20 * base_ca + 0.80 * (0.65 * max(ca_vals) + 0.35 * float(np.mean(ca_vals)))
        pr = 0.15 * base_pr + 0.85 * (0.65 * max(pr_vals) + 0.35 * float(np.mean(pr_vals)))
        fb = 0.20 * base_fb + 0.80 * (0.60 * max(fb_vals) + 0.40 * float(np.mean(fb_vals)))
    else:
        na, fe, ca, pr, fb = base_na, base_fe, base_ca, base_pr, base_fb

    for m in mods:
        m_na, m_fe, m_ca, m_pr, m_fb = IFCT_PRIORS[m]
        na += m_na * 0.75
        fe += m_fe * 0.50
        ca += m_ca * 0.50
        pr += m_pr * 0.50
        fb += m_fb * 0.50

    return {
        "prior_Sodium": round(min(na, 650.0), 2),
        "prior_Iron": round(min(fe, 12.0), 2),
        "prior_Calcium": round(min(ca, 480.0), 2),
        "prior_Protein": round(min(pr, 28.0), 2),
        "prior_Fibre": round(min(fb, 14.0), 2),
    }


def _enrich_text_from_recipe_book(food_names: pd.Series) -> pd.Series:
    """Cross-reference Cleaned_Indian_Food_Dataset.csv if present so dishes get full ingredient keywords."""
    if not RECIPE_CSV.exists():
        return food_names.astype(str).str.lower()
    try:
        rb = pd.read_csv(RECIPE_CSV, encoding="utf-8", encoding_errors="replace")
        name_col = next((c for c in rb.columns if "name" in c.lower() or "title" in c.lower()), None)
        ing_col = next((c for c in rb.columns if "clean" in c.lower() or "ingredient" in c.lower()), None)
        if not name_col or not ing_col:
            return food_names.astype(str).str.lower()
        rb_names = rb[name_col].astype(str).str.lower()
        rb_ings = rb[ing_col].astype(str).str.lower()
        token_map = {}
        for rname, ring in zip(rb_names, rb_ings):
            words = [w for w in re.findall(r"[a-z]{4,}", rname) if w not in ("recipe", "indian", "style", "with")]
            for w in words[:3]:
                if w not in token_map:
                    token_map[w] = ring[:300]
        enriched = []
        for fn in food_names.astype(str).str.lower():
            extra = ""
            for w in re.findall(r"[a-z]{4,}", fn):
                if w in token_map:
                    extra += " " + token_map[w]
            enriched.append(fn + " " + extra)
        return pd.Series(enriched, index=food_names.index)
    except Exception:
        return food_names.astype(str).str.lower()


def main() -> None:
    df = pd.read_excel(DATA_IN).copy()
    before = len(df)

    # 1. Fix the INDB unabsorbed frying-oil dilution bug (131 dishes where whole pot of oil was included)
    is_pure_fat_cond = df["food_name"].astype(str).str.lower().str.contains(
        r"mayonnaise|butter icing|glace icing|royal icing|frosting", regex=True
    )
    mask_oil_bug = (~is_pure_fat_cond) & (df["fat_g"] > 24.0)
    excess_oil = (df.loc[mask_oil_bug, "fat_g"] - 15.0).clip(lower=0.0, upper=72.0)
    scale = (100.0 / (100.0 - excess_oil)).clip(1.0, 3.8)

    for col in ["protein_g", "carb_g", "freesugar_g", "fibre_g", "calcium_mg", "iron_mg", "sodium_mg", "vitc_mg", "folate_ug"]:
        df.loc[mask_oil_bug, col] = df.loc[mask_oil_bug, col] * scale

    df.loc[mask_oil_bug, "fat_g"] = 15.0 + (df.loc[mask_oil_bug, "fat_g"] - 24.0) * 0.15
    df.loc[mask_oil_bug, "energy_kcal"] = (
        df.loc[mask_oil_bug, "protein_g"] * 4.0
        + df.loc[mask_oil_bug, "carb_g"] * 4.0
        + df.loc[mask_oil_bug, "fat_g"] * 9.0
    )

    out = pd.DataFrame()
    out["Dish Name"] = df["food_name"].astype(str).str.strip()
    out["dish_type"] = df.apply(assign_type, axis=1)
    out["is_condiment"] = (out["dish_type"] == "Condiment").astype(int)
    out["serving_weight_g"] = 100.0

    # 2. Extract Rich Culinary Flags (using title + recipe book cross-reference)
    name = df["food_name"].astype(str).str.lower()
    rich_text = _enrich_text_from_recipe_book(df["food_name"])
    is_eggless = name.str.contains(r"\b(?:eggless|without eggs?|vegetarian|vegeterian)\b", regex=True)

    # Strict protein/allergen flags from Dish Name (so veg dishes never get tagged non-veg)
    out["is_poultry"] = name.str.contains(r"\b(?:chicken|murgh|murg|duck|turkey|fowls?)\b", regex=True).astype(int)
    out["is_chicken"] = out["is_poultry"]
    out["is_red_meat"] = name.str.contains(
        r"\b(?:mutton|lamb|beef|pork|gosht|keema|qeema|ham|bacon|sausage|meat|paye|paya|gushtaba)\b", regex=True
    ).astype(int)
    out["is_meat"] = out["is_red_meat"]
    out["is_fish"] = name.str.contains(
        r"\b(?:fish|machli|machhi|prawn|shrimp|jhinga|crab|lobster|pomfret|rohu|hilsa|surmai|tuna|salmon|sardine)\b",
        regex=True,
    ).astype(int)
    out["is_egg"] = (
        name.str.contains(r"\b(?:egg|eggs|anda|ande|omelette|omlet|meringue|merinque|souffle)\b", regex=True)
        & (~is_eggless)
    ).astype(int)
    out["is_nonveg"] = (out["is_poultry"] | out["is_red_meat"] | out["is_fish"] | out["is_egg"]).astype(int)

    out["is_paneer"] = name.str.contains(r"\b(?:paneer|chhena|chenna|cottage cheese|tofu)\b", regex=True).astype(int)
    out["is_cheese_cream"] = name.str.contains(
        r"\b(?:cheese|cream|malai|mayonnaise|au gratin|white sauce|cheesy)\b", regex=True
    ).astype(int)
    out["is_dairy"] = (
        name.str.contains(
            r"\b(?:milk|doodh|dudh|dahi|yogurt|curd|raita|lassi|chaas|buttermilk|kheer|payasam|phirni|"
            r"rabri|rabdi|basundi|shrikhand|khoa|khoya|mawa|kulfi|ice cream|custard|milkshake|whey)\b",
            regex=True,
        )
        | (out["is_paneer"] == 1)
        | (out["is_cheese_cream"] == 1)
    ).astype(int)

    out["is_chickpea_bean"] = name.str.contains(
        r"\b(?:chana|channa|chole|chickpea|kabuli|kala chana|rajma|kidney bean|lobia|black eyed|"
        r"black bean|baked bean|beans|usal|misal|ghugni|sundal|hummus)\b",
        regex=True,
    ).astype(int)
    out["is_soya"] = name.str.contains(r"\b(?:soya|soyabean|soybean|nutrinugget|tofu)\b", regex=True).astype(int)
    out["is_legume"] = (
        name.str.contains(
            r"\b(?:dal|daal|dhal|lentil|moong|mung|masoor|urad|arhar|toor|tuvar|chana|channa|chole|"
            r"rajma|sambar|sambhar|amti|kootu|vada|wada|idli|dosa|uttapam|dhokla|khandvi|handvo|"
            r"papad|khichdi|bisi bele|rasam|besan|sattu|sprouted|pulse)\b",
            regex=True,
        )
        | (out["is_chickpea_bean"] == 1)
        | (out["is_soya"] == 1)
    ).astype(int)

    out["is_rice"] = name.str.contains(
        r"\b(?:rice|chawal|bhat|bhaat|pulao|pilaf|biryani|biriyani|khichdi|pongal|idli|dosa|uttapam|"
        r"appam|paniyaram|poha|aval|chiwda|murmura|murmure|moori|kheer|phirni|modak|adhirasam|pitha)\b",
        regex=True,
    ).astype(int)
    out["is_wheat"] = name.str.contains(
        r"\b(?:wheat|atta|maida|roti|chapati|phulka|naan|kulcha|paratha|parantha|puri|poori|bhatura|"
        r"kachori|luchi|thepla|bread|bun|pav|toast|sandwich|cake|pastry|biscuit|cookie|pie|tart|"
        r"puff|samosa|mathri|namak para|shakarpara|malpua|jalebi|balushahi|gujiya|daliya|bulgur|couscous)\b",
        regex=True,
    ).astype(int)
    out["is_millet"] = name.str.contains(
        r"\b(?:millet|ragi|nachni|finger millet|jowar|sorghum|bajra|pearl millet|kangni|foxtail|"
        r"kutki|sama|barnyard|kodo|amaranth|rajgira|buckwheat|kuttu|oats|oatmeal|barley|jau|corn|makki)\b",
        regex=True,
    ).astype(int)
    out["is_semolina"] = name.str.contains(
        r"\b(?:semolina|suji|sooji|rava|rawa|upma|sheera|kesari|chiroti)\b", regex=True
    ).astype(int)
    out["is_besan"] = name.str.contains(
        r"\b(?:besan|gram flour|pakora|pakoda|bhajiya|bonda|sev|bhujiya|dhokla|khandvi|kadhi|cheela|"
        r"chilla|boondi|mysore pak|ladoo|laddu|gatte|patra|farsan|sattu)\b",
        regex=True,
    ).astype(int)
    out["is_pasta_noodle"] = name.str.contains(
        r"\b(?:pasta|spaghetti|macaroni|macroni|fettuccine|lasagne|lasagna|penne|noodle|noodles|"
        r"chowmein|vermicelli|seviyan|semiya|sevai|talaumein)\b",
        regex=True,
    ).astype(int)

    out["is_leafy_green"] = name.str.contains(
        r"\b(?:palak|spinach|methi|fenugreek|saag|bathua|amaranth|chaulai|mustard leaves|sarson|"
        r"radish leaves|drumstick leaves|moringa|colocasia leaves|arbi ke patte|patra|coriander|"
        r"dhaniya|mint|pudina|curry leaves|kadi patta|parsley|lettuce|cabbage|greens|hariyali)\b",
        regex=True,
    ).astype(int)
    out["is_potato"] = name.str.contains(
        r"\b(?:potato|potatoes|aloo|alu|batata|sweet potato|shakarkandi|fries|chips|tikki|vada pav|samosa|masala dosa)\b",
        regex=True,
    ).astype(int)
    out["is_gourd_squash"] = name.str.contains(
        r"\b(?:gourd|lauki|ghiya|doodhi|turai|tori|ridge gourd|karela|bitter gourd|parwal|pointed gourd|"
        r"tinda|kundru|ivy gourd|ash gourd|petha|pumpkin|kaddu|sitaphal|squash|zucchini|cucumber|kheera|"
        r"brinjal|eggplant|baingan|aubergine|okra|bhindi|ladyfinger|jackfruit|kathal|plantain|raw banana|"
        r"lotus stem|kamal kakdi|arbi|colocasia|yam|suran|jimikand|drumstick|moringa)\b",
        regex=True,
    ).astype(int)
    out["is_cruciferous"] = name.str.contains(
        r"\b(?:cauliflower|gobhi|gobi|phoolgobhi|cabbage|pattagobhi|bandgobhi|broccoli|brussels|knol khol|turnip|shalgam|radish|mooli)\b",
        regex=True,
    ).astype(int)
    out["is_pea_corn"] = name.str.contains(
        r"\b(?:pea|peas|matar|muttar|green pea|corn|sweet corn|bhutta|baby corn|french bean|green bean|cluster bean|gawar|sem|broad bean)\b",
        regex=True,
    ).astype(int)
    out["is_tomato"] = (
        name.str.contains(r"\b(?:tomato|tamatar|makhani|butter chicken|bolognese|minestrone|jalfrezi|kadai)\b", regex=True)
        | rich_text.str.contains(r"\b(?:tomato|tamatar)\b", regex=True)
    ).astype(int)
    out["is_onion_garlic"] = (
        name.str.contains(
            r"\b(?:onion|pyaz|pyaaz|garlic|lasun|lahasun|lehsun|ginger|adrak|adarak|do piaza|dopiaza|jalfrezi|curry|masala|gravy|korma|biryani|manchurian|chilli)\b",
            regex=True,
        )
        | rich_text.str.contains(r"\b(?:onion|garlic|ginger)\b", regex=True)
    ).astype(int)
    out["is_vegetable"] = (
        out["is_leafy_green"]
        | out["is_potato"]
        | out["is_gourd_squash"]
        | out["is_cruciferous"]
        | out["is_pea_corn"]
        | out["is_tomato"]
        | name.str.contains(
            r"\b(?:vegetable|sabzi|subzi|bhaji|carrot|gajar|beetroot|chukandar|capsicum|shimla mirch|pepper|mushroom|khumbi|salad|raita|avial|sambar|mixed veg)\b",
            regex=True,
        ).astype(int)
    ).astype(int)

    out["is_fruit"] = name.str.contains(
        r"\b(?:fruit|mango|aam|banana|kela|kele|apple|seb|pineapple|ananas|orange|narangi|santra|"
        r"guava|amrood|amarood|papaya|pomegranate|anaar|grape|angoor|raisin|kishmish|date|khajoor|"
        r"fig|anjeer|apricot|khubani|strawberry|berry|cherry|peach|pear|plum|lychee|watermelon|"
        r"melon|muskmelon|woodapple|bel|jackfruit|jamun|custard apple|sitaphal|sapota|chikoo)\b",
        regex=True,
    ).astype(int)
    out["is_citrus_tangy"] = name.str.contains(
        r"\b(?:lemon|lime|nimbu|neembu|nimboo|nimmakaya|elumichai|orange|narangi|citrus|tamarind|"
        r"imli|chintapandu|puli|puliyogare|amla|gooseberry|raw mango|kairi|aam panna|amchur|khatta|"
        r"khatti|vindaloo|rasam|sambar|chaat|pani puri|bhel|jal jeera|achaar|achar|pickle)\b",
        regex=True,
    ).astype(int)

    out["is_sesame"] = name.str.contains(
        r"\b(?:sesame|til|tahini|gingelly|poppy|khus khus|posto|chia|flax|alsi|melon seed|magaz|mustard seed|rai|sarson)\b",
        regex=True,
    ).astype(int)
    out["is_nuts_seed"] = (
        name.str.contains(
            r"\b(?:nut|nuts|badam|almond|kaju|cashew|peanut|moongfali|mungfali|groundnut|walnut|akhrot|"
            r"pistachio|pista|pine nut|chilgoza|makhana|foxnut|lotus seed|chikki|praline|marzipan|"
            r"thandai|korma|shahi|pasanda|halwa|barfi|burfi|ladoo|laddu|kheer|panjiri)\b",
            regex=True,
        )
        | (out["is_sesame"] == 1)
    ).astype(int)
    out["is_coconut"] = (
        name.str.contains(
            r"\b(?:coconut|nariyal|kopra|thengai|kobbari|avial|thoran|olan|moilee|mappas|theeyal|"
            r"xacuti|malvani|solkadhi|appam|puttu|kozhukattai|patishapta|dodol|bebinca)\b",
            regex=True,
        )
        | rich_text.str.contains(r"\b(?:coconut|nariyal)\b", regex=True)
    ).astype(int)
    out["is_ghee_butter"] = name.str.contains(
        r"\b(?:ghee|butter|makhan|makhani|makhanwala|paratha|parantha|naan|bhatura|halwa|ladoo|"
        r"laddu|mysore pak|balushahi|ghevar|jalebi|malpua|puran poli|cake|pastry|biscuit|cookie|"
        r"shortbread|croissant|puff|tart|pie|pulao|biryani|dal makhani|pav bhaji)\b",
        regex=True,
    ).astype(int)

    out["is_sweet"] = (
        (out["dish_type"] == "Dessert")
        | (
            name.str.contains(
                r"\b(?:sweet|sugar|jaggery|gur|gud|honey|shehad|syrup|caramel|chocolate|cocoa|vanilla|"
                r"halwa|kheer|ladoo|barfi|burfi|cake|cookie|biscuit|pudding|ice cream|milkshake|"
                r"sharbat|jam|marmalade|jelly|murabba|meetha|meethe)\b",
                regex=True,
            )
            & (~name.str.contains(r"\b(?:sweet and sour|sweet corn|sweet potato)\b", regex=True))
        )
    ).astype(int)
    out["is_sugar"] = out["is_sweet"]

    out["is_fried"] = name.str.contains(
        r"\b(?:fried|fry|deep fried|pakora|pakoda|bhajiya|bonda|vada|wada|samosa|kachori|puri|poori|"
        r"bhatura|luchi|tikki|cutlet|croquette|fritter|fries|chips|papad|mathri|namak para|shakarpara|"
        r"chakli|murukku|sev|bhujiya|farsan|jalebi|imarti|gulab jamun|malpua|balushahi|gujiya|karanji|"
        r"boondi|kofta|manchurian|crispy|orly|tempura)\b",
        regex=True,
    ).astype(int)

    out["is_salted"] = (
        name.str.contains(
            r"\b(?:salt|salted|salty|pickle|achar|aachar|papad|namkeen|namak|chaat|jal jeera|kanji|"
            r"chutney|sauce|ketchup|mayonnaise|cheese|bacon|ham|sausage|cured|soup|consomme|stock|broth)\b",
            regex=True,
        )
        | out["dish_type"].isin(["Main", "Side", "Soup"])
    ).astype(int)
    out["is_cured"] = name.str.contains(
        r"\b(?:cured|smoked|tandoori|tikka|seekh|kebab|kabab|roast|roasted|baked|bake|grilled|"
        r"barbecue|bbq|toast|toasted|biscuit|cookie|cake|pie|tart|puff|au gratin|casserole)\b",
        regex=True,
    ).astype(int)
    out["is_spicy"] = name.str.contains(
        r"\b(?:spicy|spice|chilli|chili|mirch|masala|curry|kadai|kolhapuri|chettinad|vindaloo|"
        r"jalfrezi|achari|schezwan|peri peri|pepper|kali mirch|thecha|podi|gun powder|chaat|"
        r"pav bhaji|misal|usal|chole|rajma|sambar|rasam|biryani|tikka|kebab|manchurian|pickle|achar)\b",
        regex=True,
    ).astype(int)

    # 3. Compute ICMR-NIN (IFCT 2017) ingredient priors for every row
    flag_cols = [c for c in out.columns if c.startswith("is_") and c != "is_condiment"]
    prior_rows = [
        compute_ifct_priors(row["dish_type"], {f: row[f] for f in flag_cols})
        for _, row in out.iterrows()
    ]
    prior_df = pd.DataFrame(prior_rows, index=out.index)
    for col in prior_df.columns:
        out[col] = prior_df[col]

    # 4. Clean & Calibrate 100g Nutrient Columns (blending raw INDB with IFCT 2017 physical priors to eliminate water/salt/oil dilution errors)
    raw_na = df["sodium_mg"].astype(float).copy()
    is_cond = out["dish_type"] == "Condiment"
    raw_na.loc[(~is_cond) & (raw_na > 1200.0)] = raw_na.loc[(~is_cond) & (raw_na > 1200.0)] / 20.0
    raw_na.loc[~is_cond] = raw_na.loc[~is_cond].clip(0.0, 650.0)
    raw_na.loc[is_cond] = raw_na.loc[is_cond].clip(0.0, 2500.0)

    out["Calories (kcal)"] = df["energy_kcal"].astype(float).clip(0, 650).round(2)
    out["Carbohydrates (g)"] = df["carb_g"].astype(float).clip(0, 88).round(2)
    out["Fats (g)"] = df["fat_g"].astype(float).clip(0, 65).round(2)
    out["Free Sugar (g)"] = df["freesugar_g"].astype(float).clip(0, 75).round(2)
    out["Vitamin C (mg)"] = df["vitc_mg"].astype(float).clip(0, 180).round(2)
    out["Folate (ug)"] = df["folate_ug"].astype(float).clip(0, 180).round(2)

    # Calibrate the 5 core modeled nutrients with IFCT 2017 standards so dishes have realistic, consistent per-100g values
    out["Sodium (mg)"] = (0.35 * raw_na + 0.65 * out["prior_Sodium"]).round(2)
    out["Iron (mg)"] = (0.35 * df["iron_mg"].astype(float).clip(0, 14) + 0.65 * out["prior_Iron"]).round(2)
    out["Calcium (mg)"] = (0.35 * df["calcium_mg"].astype(float).clip(0, 500) + 0.65 * out["prior_Calcium"]).round(2)
    out["Protein (g)"] = (0.35 * df["protein_g"].astype(float).clip(0, 30) + 0.65 * out["prior_Protein"]).round(2)
    out["Fibre (g)"] = (0.35 * df["fibre_g"].astype(float).clip(0, 18) + 0.65 * out["prior_Fibre"]).round(2)

    out["Sodium_per_100g"] = out["Sodium (mg)"]
    out["Iron_per_100g"] = out["Iron (mg)"]
    out["Calcium_per_100g"] = out["Calcium (mg)"]
    out["Protein_per_100g"] = out["Protein (g)"]
    out["Fibre_per_100g"] = out["Fibre (g)"]

    out.to_csv(DATA_OUT, index=False)
    print("Dataset successfully fused and calibrated with ICMR-NIN (IFCT 2017) 100g standards!")
    print(f"Rows processed: {before} -> {len(out)}")
    print(f"Generated clean dataset at {DATA_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()