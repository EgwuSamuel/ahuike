"""Catalogue of clinical findings used by the AHỤIKE protocol engine.

Each finding carries the surface text used to render case vignettes in
Nigerian English. Labels never come from this text: they come from
`rules.py`, which reads only the structured case.

Voices:
  child    - a caregiver describing a child/infant; uses {He} {he} {his} {His} {him}
  self     - a woman describing herself (pregnant / postpartum)
  partner  - a husband or relative describing her ("she")
  clinical - a short phrase for a CHEW clinical note
`{dur}` is replaced with a duration phrase ("3 days", "2 weeks").

Sources (simplified for triage, see docs/ETHICS.md):
  WHO IMCI Chart Booklet (2014); WHO IMNCI young infant charts;
  Nigeria National Standing Orders for CHEWs/JCHEWs (maternal danger signs);
  WHO "Pregnancy, Childbirth, Postpartum and Newborn Care" (danger signs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

CHILD = "child"          # 2-59 months
INFANT = "infant"        # 0-59 days
PREGNANT = "pregnant"
POSTPARTUM = "postpartum"
POPULATIONS = (CHILD, INFANT, PREGNANT, POSTPARTUM)


@dataclass(frozen=True)
class Finding:
    id: str
    populations: tuple[str, ...]
    clinical: str
    child: tuple[str, ...] = ()
    self_: tuple[str, ...] = ()
    partner: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()   # at least one of these must also be present
    excludes: tuple[str, ...] = ()   # cannot co-occur with these
    min_ga: int = 0                  # minimum gestational age (weeks) for pregnant cases
    has_duration: bool = False
    weight: float = 1.0              # sampling weight
    tags: tuple[str, ...] = field(default=())


F = Finding

CATALOGUE: tuple[Finding, ...] = (
    # ---------------------------------------------------------------- child (IMCI 2-59 months)
    F("unable_to_drink", (CHILD,), "unable to drink or breastfeed",
      child=("{He} cannot drink or breastfeed at all.",
             "{He} refuses the breast and cannot even take water.",
             "Since morning {he} has not been able to suck or drink anything."),
      excludes=("drinking_poorly", "drinks_eagerly")),
    F("drinking_poorly", (CHILD,), "drinking poorly",
      child=("{He} is drinking very little, only small sips.",
             "{He} is not drinking well, {he} only takes a little water."),
      requires=("diarrhoea",), excludes=("unable_to_drink", "drinks_eagerly")),
    F("vomits_everything", (CHILD,), "vomits everything",
      child=("{He} vomits everything {he} eats or drinks.",
             "Anything we give {him}, even water, {he} throws it up."),
      weight=0.8),
    F("convulsions", (CHILD, INFANT, PREGNANT, POSTPARTUM), "history of convulsions",
      child=("{He} had convulsion yesterday, {his} body was jerking.",
             "{His} body was shaking and {his} eyes rolled up, like convulsion.",
             "{He} started jerking and stiffening this morning."),
      self_=("I had convulsion last night, my body was jerking.",
             "I had fits this morning and I fell down."),
      partner=("She had convulsion, her body was jerking and she fell down.",
               "She had fits this morning and did not know herself for some minutes."),
      weight=0.8),
    F("lethargic", (CHILD,), "lethargic / unconscious",
      child=("{He} is very weak and does not respond when we call {him}.",
             "{He} is just lying down, too weak, {he} is not reacting like before.",
             "We cannot wake {him} properly, {he} goes unconscious on and off."),
      excludes=("restless_irritable",), weight=0.8),
    F("cough", (CHILD,), "cough x {dur}",
      child=("{He} has been coughing for {dur}.",
             "{He} has had cough for {dur} now."),
      has_duration=True, weight=2.0),
    F("chest_indrawing", (CHILD, INFANT), "lower chest wall indrawing",
      child=("When {he} breathes in, the lower chest goes inside.",
             "{His} chest is pulling in under the ribs when {he} breathes."),
      weight=0.8),
    F("stridor", (CHILD,), "stridor in a calm child",
      child=("{He} makes a harsh noise when breathing in, even when calm.",
             "There is a noisy, harsh sound when {he} breathes in while resting."),
      weight=0.5),
    F("fast_breathing", (CHILD, INFANT), "fast breathing",
      child=("{He} is breathing very fast.",
             "{His} breathing is fast, faster than normal."),
      weight=1.5, tags=("vital_rr",)),
    F("diarrhoea", (CHILD,), "diarrhoea x {dur}",
      child=("{He} has running stomach for {dur}.",
             "{He} has been passing watery stool for {dur}.",
             "{He} has had diarrhoea for {dur}."),
      has_duration=True, weight=2.0),
    F("sunken_eyes", (CHILD,), "sunken eyes",
      child=("{His} eyes have sunk inside.", "{His} eyes look sunken."),
      requires=("diarrhoea",), weight=1.2),
    F("skin_pinch_very_slow", (CHILD,), "skin pinch goes back very slowly (>2 s)",
      child=("When I pinch {his} belly skin it stays up for long before going down.",
             "The skin of {his} stomach goes back very slowly when pinched."),
      requires=("diarrhoea",), excludes=("skin_pinch_slow",)),
    F("skin_pinch_slow", (CHILD,), "skin pinch goes back slowly",
      child=("When pinched, {his} skin goes back a bit slowly.",),
      requires=("diarrhoea",), excludes=("skin_pinch_very_slow",)),
    F("restless_irritable", (CHILD,), "restless, irritable",
      child=("{He} is restless and crying, {he} cannot settle.",
             "{He} is irritable and keeps crying."),
      requires=("diarrhoea",), excludes=("lethargic",)),
    F("drinks_eagerly", (CHILD,), "drinks eagerly, thirsty",
      child=("{He} is very thirsty, {he} drinks water eagerly.",
             "{He} keeps asking for water and drinks it fast."),
      requires=("diarrhoea",), excludes=("unable_to_drink", "drinking_poorly")),
    F("blood_in_stool", (CHILD,), "blood in stool",
      child=("There is blood in {his} stool.", "{He} is passing stool with blood."),
      requires=("diarrhoea",), weight=0.7),
    F("fever", (CHILD, INFANT, PREGNANT, POSTPARTUM), "fever x {dur}",
      child=("{He} has had hot body for {dur}.",
             "{He} has had fever for {dur}.",
             "{His} body has been hot for {dur}."),
      self_=("I have had fever for {dur}.", "My body has been hot for {dur}."),
      partner=("She has had fever for {dur}.", "Her body has been hot for {dur}."),
      has_duration=True, weight=2.0),
    F("stiff_neck", (CHILD,), "stiff neck",
      child=("{His} neck is stiff, {he} cannot bend it.",
             "{He} cannot bend {his} neck forward, it is stiff."),
      requires=("fever",), weight=0.6),
    F("oedema_feet", (CHILD,), "bilateral pitting oedema of feet",
      child=("Both of {his} feet are swollen.",
             "{His} two feet are swollen and when I press, it leaves a dent."),
      weight=0.5),
    F("runny_nose", (CHILD, INFANT), "rhinorrhoea",
      child=("{He} has catarrh.", "{He} has catarrh and is sneezing.",
             "{His} nose is running."),
      weight=1.5),
    F("teething", (CHILD,), "teething, drooling",
      child=("{He} is teething and drooling.", "{His} teeth are coming out and {he} is drooling."),
      weight=1.0),
    F("mild_rash", (CHILD,), "mild heat rash",
      child=("{He} has small heat rash on {his} neck.",),
      weight=0.7),

    # ---------------------------------------------------------------- young infant (0-59 days)
    F("not_feeding_well", (INFANT,), "not feeding well",
      child=("{He} is not feeding well, {he} refuses the breast.",
             "{He} stopped sucking well since yesterday.")),
    F("low_temperature", (INFANT,), "low body temperature",
      child=("{His} body is cold to touch.", "{He} feels cold even when wrapped."),
      excludes=("fever",), tags=("vital_temp",)),
    F("movement_only_when_stimulated", (INFANT,), "moves only when stimulated",
      child=("{He} does not move unless we touch {him}.",
             "{He} is not moving at all, only when we shake {him}.")),
    F("jaundice_palms_soles", (INFANT,), "jaundice of palms and soles",
      child=("{His} eyes are yellow and even {his} palms and soles are yellow.",
             "The yellow colour has reached {his} palms and feet.")),
    F("umbilicus_pus", (INFANT,), "umbilicus red / draining pus",
      child=("{His} navel is red and has pus.",
             "The cord area is red and there is pus coming out.")),
    F("skin_pustules", (INFANT,), "skin pustules",
      child=("{He} has small pus bumps on {his} skin.",)),
    F("milk_spit_up", (INFANT,), "small spit-up after feeds",
      child=("{He} brings out small milk after feeding.",
             "{He} spits up a little milk after breastfeeding."),
      weight=1.5),
    F("blocked_nose", (INFANT,), "blocked nose",
      child=("{His} nose is blocked and {he} sneezes sometimes.",),
      weight=1.2),

    # ---------------------------------------------------------------- pregnancy & postpartum
    F("vaginal_bleeding", (PREGNANT,), "vaginal bleeding",
      self_=("I am bleeding from my private part.",
             "I saw blood coming from my vagina this morning."),
      partner=("She is bleeding from her private part.",
               "She saw blood coming from her vagina this morning.")),
    F("heavy_bleeding", (POSTPARTUM,), "heavy postpartum bleeding",
      self_=("I am bleeding heavily since I delivered, I soak a pad in a few minutes.",
             "The bleeding after delivery is too much, blood is soaking everything."),
      partner=("She is bleeding heavily since delivery, she soaks a pad in a few minutes.",),
      excludes=("normal_lochia",)),
    F("severe_headache", (PREGNANT, POSTPARTUM), "severe headache",
      self_=("I have a very severe headache that will not go.",
             "My head is aching me seriously, paracetamol did not help."),
      partner=("She has a very severe headache that is not going.",)),
    F("blurred_vision", (PREGNANT, POSTPARTUM), "blurred vision",
      self_=("My eyes are blurred, I cannot see clearly.",
             "I am seeing things blurry, like stars."),
      partner=("She says her eyes are blurred and she cannot see well.",)),
    F("face_hand_swelling", (PREGNANT, POSTPARTUM), "swelling of face and hands",
      self_=("My face and hands are swollen.", "My face is puffy and my fingers are swollen."),
      partner=("Her face and hands are swollen.",)),
    F("severe_abdominal_pain", (PREGNANT, POSTPARTUM), "severe abdominal pain",
      self_=("I have serious pain in my stomach that is not stopping.",),
      partner=("She has serious stomach pain that is not stopping.",), weight=0.8),
    F("difficulty_breathing", (PREGNANT, POSTPARTUM), "difficulty breathing",
      self_=("I am finding it hard to breathe.", "I am breathing with difficulty even when resting."),
      partner=("She is finding it hard to breathe.",), weight=0.7),
    F("water_broke", (PREGNANT,), "leaking amniotic fluid before labour",
      self_=("Water is leaking from my private part but labour has not started.",),
      partner=("Water is leaking from her private part but labour has not started.",),
      min_ga=20, weight=0.7),
    F("no_fetal_movement", (PREGNANT,), "no fetal movement",
      self_=("I have not felt the baby move since yesterday.",),
      partner=("She has not felt the baby move since yesterday.",),
      excludes=("reduced_fetal_movement",), min_ga=28, weight=0.7),
    F("reduced_fetal_movement", (PREGNANT,), "reduced fetal movement",
      self_=("The baby is moving less than usual.",),
      partner=("She says the baby is moving less than usual.",),
      excludes=("no_fetal_movement",), min_ga=28),
    F("high_fever_very_weak", (PREGNANT, POSTPARTUM), "high fever, too weak to get up",
      self_=("I have high fever and I am too weak to get out of bed.",),
      partner=("She has high fever and is too weak to get out of bed.",),
      excludes=("fever",), weight=0.6),
    F("prolonged_labour", (PREGNANT,), "labour > 12 hours",
      self_=("My labour pain started more than 12 hours ago and the baby has not come.",),
      partner=("Her labour pain started more than 12 hours ago and the baby has not come.",),
      min_ga=34, weight=0.6),
    F("vomiting_cannot_keep", (PREGNANT,), "vomiting, cannot keep fluids down",
      self_=("I am vomiting everything, I cannot keep food or water down.",),
      partner=("She is vomiting everything, she cannot keep food or water down.",),
      excludes=("mild_nausea",)),
    F("burning_urination", (PREGNANT, POSTPARTUM), "dysuria",
      self_=("It burns when I urinate.", "I feel pain and burning when passing urine."),
      partner=("She says it burns when she urinates.",)),
    F("foul_discharge", (POSTPARTUM,), "foul-smelling vaginal discharge",
      self_=("The discharge from my private part is smelling bad.",),
      partner=("The discharge from her private part is smelling bad.",)),
    F("painful_red_breast", (POSTPARTUM,), "painful, red, hot breast",
      self_=("My breast is painful, red and hot.",),
      partner=("Her breast is painful, red and hot.",)),
    F("low_mood", (POSTPARTUM,), "low mood, not coping",
      self_=("I feel very sad and I cannot cope with the baby.",),
      partner=("She is always crying and says she cannot cope with the baby.",)),
    F("mild_nausea", (PREGNANT,), "mild morning nausea",
      self_=("I feel like vomiting in the morning but I can eat.",),
      partner=("She feels like vomiting in the morning but she can eat.",),
      excludes=("vomiting_cannot_keep",), weight=1.3),
    F("back_pain", (PREGNANT,), "mild back pain",
      self_=("My back is aching me small.", "I have mild back pain."),
      partner=("She has mild back pain.",), weight=1.3),
    F("heartburn", (PREGNANT,), "heartburn",
      self_=("I have heartburn after eating.",),
      partner=("She has heartburn after eating.",), weight=1.2),
    F("leg_cramps", (PREGNANT,), "leg cramps at night",
      self_=("I have leg cramps at night.",),
      partner=("She has leg cramps at night.",), weight=1.2),
    F("feet_swelling_evening", (PREGNANT,), "mild ankle swelling in the evening",
      self_=("My feet swell small in the evening but go down by morning.",),
      partner=("Her feet swell small in the evening but go down by morning.",), weight=1.0),
    F("tiredness", (PREGNANT, POSTPARTUM), "tiredness",
      self_=("I feel tired most of the time.",),
      partner=("She feels tired most of the time.",), weight=1.3),
    F("after_pains", (POSTPARTUM,), "mild after-pains",
      self_=("I have mild cramps in my belly when breastfeeding.",),
      partner=("She has mild belly cramps when breastfeeding.",), weight=1.2),
    F("sore_nipples", (POSTPARTUM,), "sore nipples",
      self_=("My nipples are sore.",),
      partner=("Her nipples are sore.",), weight=1.2),
    F("normal_lochia", (POSTPARTUM,), "light lochia",
      self_=("I still have small bleeding, like a light period.",),
      partner=("She still has small bleeding, like a light period.",),
      excludes=("heavy_bleeding",), weight=1.2),
)

BY_ID: dict[str, Finding] = {f.id: f for f in CATALOGUE}


def for_population(population: str) -> list[Finding]:
    return [f for f in CATALOGUE if population in f.populations]
