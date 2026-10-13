from src import lexicon


def test_fold_strips_accents():
    assert lexicon.fold("Recuperación NÃO") == "recuperacion nao"


def test_spanish_negative_and_positive():
    assert lexicon.score_text("Huracán deja dos muertos y daños graves", "es")["compound"] < -0.5
    assert lexicon.score_text("La economía crece y se recupera el empleo", "es")["compound"] > 0.3


def test_spanish_negation_flips_valence():
    plain = lexicon.score_text("hubo víctimas", "es")["compound"]
    negated = lexicon.score_text("no hubo víctimas", "es")["compound"]
    assert plain < 0 < negated


def test_portuguese_negation_and_scores():
    assert lexicon.score_text("Enchentes deixam mortos", "pt")["compound"] < -0.5
    assert lexicon.score_text("não houve mortos", "pt")["compound"] > 0


def test_intensifier_increases_magnitude():
    base = lexicon.score_text("resultado malo", "es")["compound"]
    boosted = lexicon.score_text("resultado muy malo", "es")["compound"]
    assert boosted < base < 0


def test_neutral_text_scores_zero():
    out = lexicon.score_text("El gobierno publica el calendario de audiencias", "es")
    assert out["compound"] == 0.0 and out["matches"] == []


def test_compound_is_bounded():
    text = " ".join(["muerte tragedia masacre"] * 20)
    assert -1.0 <= lexicon.score_text(text, "es")["compound"] <= -0.9


def test_compositional_bigrams():
    assert lexicon.score_text("Desemprego cai no trimestre", "pt")["compound"] > 0
    assert lexicon.score_text("Cae el desempleo en octubre", "es")["compound"] > 0


def test_death_verbs():
    assert lexicon.score_text("Muere actor a los 34 años", "es")["compound"] < -0.3
    assert lexicon.score_text("Morre ator aos 34 anos", "pt")["compound"] < -0.3
