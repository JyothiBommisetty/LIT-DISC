# Narrative Event Log

**Novel:** The Mistress of Spices (Chitra Banerjee Divakaruni)  
**Processing:** 105 windows of ~800 words, LIT-DISC v1.1 (sigma_max = 5.0)

---

## Summary of detected events

| Event | Window | Cumulative words | LDRI Δ | Peak LDRI | Narrative trigger |
|-------|--------|----------------|--------|-----------|-------------------|
| E1 | W3→W4 | 2,327→3,066 | **−0.0067** | — | Shift from narrator's proud youth to prosaic arranged-marriage conversation |
| E2 | W14 | ~10,899 | Valley (0.0067) | — | Early Oakland store scene — mundane customer interactions |
| E3 | W17→W18 | 13,221→14,018 | **−0.0054** | — | Channeling the Old One — anger and regret over past mistakes |
| E4 | W25→W26 | 19,390→20,142 | **+0.0078** | 0.0152 | Makaradwaj spice lore — Tilo's defiant claim "I am not afraid" |
| E5 | W34 | ~26,432 | Valley (0.0061) | — | Domestic scenes — customers' everyday struggles |
| E6 | W40 | ~31,168 | — | **0.0155** | Spiritual crisis: "Spices is this. The way. You have chosen. To. Punish me." |
| E7 | W44→W45 | 34,264→35,064 | **+0.0061** | 0.0150 | Contacting First Mother — Tilo's apology, mentor's exhaustion |
| E8 | W50 | ~38,835 | — | — | Childhood memory — mother's nightly ritual (low raw LDRI) |
| E9 | W54→W55 | 41,631→42,359 | **+0.0068** | 0.0147 | **Racist assault**: boot, pain, "sonofa-bitch Indian" → Tilo fights back |
| E10 | W70 | ~53,869 | — | — | Traumatic memory: "dark room would drop like a film over my eyes" (low raw) |
| E11 | W78 | ~60,147 | — | — | Highest intrinsic (0.121): bent woman in gray, guilt, exhaustion |
| E12 | W80 | ~61,439 | — | — | Tilo touches a customer — "burn of searing iron, prick of poison needle" |
| E13 | W81 | ~62,238 | **+0.0067** | 0.0139 | Joyful reunion: hugging, laughing, tears — "Forgive me, Grandmother" |
| E14 | W81→W82 | 62,238→63,012 | **−0.0082** | →0.0057 | **Crisis**: "BIGGEST SALE OF THE YEAR... EVERYTHING MUST GO" — Tilo breaks her vow |
| E15 | W89 | ~68,453 | — | 0.0149 | Liberation: "O you who are young" — transformation, physical freedom |
| E16 | W91 | ~70,024 | Valley (0.0072) | — | Attempting motherhood: "Why not I go see the doctor" (low raw) |

---

## Detailed event descriptions

### E1 — Youth and pride yields to the mundane (W3→W4, ΔLDRI = −0.0067)

**W3** opens with the young narrator's vanity: fine muslins, tortoiseshell combs, pride. It is lush, sensory prose — a self-portrait of a girl on the cusp of becoming a spice mistress. LDRI is elevated (0.0130).

**W4** pivots abruptly to a neighbor arranging a marriage with a boy in America. The narrator asks "Why don't you work in this country?" The language shifts from the narrator's past to present-day pragmatic dialogue. The drop in literary intensity correlates with this shift from lyrical memory to mundane transaction.

### E3 — Channeling the Old One (W17→W18, ΔLDRI = −0.0054)

W17 has elevated LDRI (0.0130) as Tilo channels the Old One's warning. W18 opens with "Anger whose other name is regret surged like wildfire through me." Yet the raw LDRI drops to its lowest point up to that window (0.920), the second-lowest in the entire novel. The anger is raw but prosaic — a first-person confessional that lacks the layered symbolic density of the peak windows.

### E4 — The defiant spice mistress (W25→W26, ΔLDRI = +0.0078, largest increase)

**W25** is a customer discussion of domestic hardship — low LDRI (0.0074).

**W26** surges to LDRI=0.0152. Tilo invokes the *makaradwaj* (rejuvenator spice), claiming she is "not afraid" of its power. The prose moves between spice lore (Ashwini Kumars, Dhanwantari) and present dilemma (a fat man speaking). The high intrinsic (z=2.3) reflects the activation of symbolic-spice vocabulary that triggers multiple feature dimensions simultaneously.

### E6 — The spiritual crisis (W40, LDRI = 0.0155, highest in novel)

This is the novel's absolute LDRI peak. The text is fragmented and incantatory:

> Spices is this. The way. You have chosen. To. Punish me. *Adrak* ginger, be with me. I drop the slices in a pan of boiling water, watch them rise and sink, rise and sink, in a slow whirl. Like lives.

The passage is extraordinary for its compression: declarative statements, direct address to spices, metaphor layered on metaphor. The intrinsic component is z=3.5 above the mean — the feature extractors detect maximal activation across semantic, narrative, symbolic, and emotion dimensions simultaneously. The runtime also spikes (1.46s, z=4.0), consistent with the density of features requiring more processing.

### E9 — The assault (W54→W55, ΔLDRI = +0.0068)

**W54** is philosophical: Tilo asks First Mother about seeing the "tapestry" of fate. LDRI is low (0.0079).

**W55** is the novel's only physical-violence scene: a racist assault on an Indian man (presumably Raven). "The blunt boot tip catches him in the armpit... sonofa-bitch Indian." The LDRI jumps to 0.0147. This is the most narratively vivid action scene in the novel — physical conflict, visceral sensation, dialogue, and internal monologue interwoven. The jump in gamma (0.0085→0.0155) suggests the feature extractors register this passage with high confidence across multiple dimensions.

### E13-E14 — The joyful reunion and the breaking vow (W81→W82)

**W81** (LDRI=0.0139, Δ+0.0067 from W80): Hugging, laughing, tears. A trick played in jest. "Forgive me, Grandmother." This is a genuine emotional peak — Tilo is reunited with someone she loves.

**W82** (LDRI=0.0057, Δ−0.0082): The novel's deepest valley. Tilo hangs signs: "BIGGEST SALE OF THE YEAR / BEST BARGAIN IN TOWN / EVERYTHING MUST GO." Raven walks in. Tilo is closing the store — breaking her vows as a spice mistress. The LDRI crashes because the emotional and symbolic register shifts from reunion (peak) to renunciation (valley). The language becomes terse, declarative, commercial — stripped of all literary richness. This is the narrative's turning point.

### E15 — Liberation (W89, LDRI = 0.0149)

> O you who are young, you will never know the delight with which I rise from bed, how the simple act of stretching these newly middleaged arms upward makes me giddy with forbidden pleasure. I shower, running my hands over my body, feeling it grow firmer even as I touch.

Tilo has transformed. Her body is changing (she is becoming human, losing her spice-mistress immunity to age). The LDRI spike reflects this passage's extraordinary sensory-somatic density: touch, sight, movement, and the dizzying freedom of physical existence. The intrinsic component is z=3.4.

---

## Windows with low raw LDRI (z < −2.3)

| Window | Cumulative words | Raw LDRI | z-score | Narrative context |
|--------|----------------|----------|---------|-------------------|
| W18 | 14,018 | 0.920 | −3.2 | Anger and regret after channeling the Old One |
| W50 | 38,835 | 0.922 | −3.1 | Childhood memory — mother's bedtime ritual |
| W54 | 41,631 | 0.921 | −3.1 | Philosophical discussion with First Mother |
| W70 | 53,869 | 0.920 | −3.2 | Traumatic memory flashback |
| W91 | 70,024 | 0.928 | −2.4 | Attempting pregnancy — doctor visit |

These windows all share a common feature: they are **first-person narrative reflection** — memory, introspection, philosophical dialogue. The raw LDRI (which captures activation richness) is lowest when the prose is most introspective and least layered with symbolic and interactional density. This aligns with the finding that raw LDRI reflects discourse-in-action more than internal reflection.

---

## Novel-level narrative arc

The LDRI trajectory across the 105 windows traces a clear narrative arc:

1. **Opening** (W1–W10): The store, the spices, Tilo's past. LDRI moderate but falling as the lyrical opening yields to present-day exposition.
2. **Early crisis** (W11–W20): Rising LDRI peaks at W17 (channeling the Old One), then plunges into dark reflection.
3. **Middle plateau** (W21–W35): LDRI oscillates in a lower range with one major spike at W26 (makaradwaj lore).
4. **Spiritual crisis** (W36–W50): Two large spikes: W40 (self-punishment through spices, the novel's peak) and W45 (seeking First Mother). Both involve direct address to the spiritual dimension of the narrative.
5. **Action sequence** (W51–W60): The assault at W55 produces the period's highest LDRI.
6. **Memory and trauma** (W61–W75): LDRI dips as Tilo processes her customers' and her own traumatic memories.
7. **Climax** (W76–W85): The reunion (W81) and the renunciation (W82) form the tightest peak-to-valley transition in the novel — a genuine literary climax where the protagonist's choice pivots the narrative.
8. **Resolution** (W86–W105): LDRI fluctuates but never recovers to the heights of the first half. W89 (transformation) is a final spike before the denouement.

This arc validates that LDRI, despite gamma dominance, tracks the novel's narrative structure. The peaks correspond to scenes of highest literary-discourse intensity: spiritual crisis, physical violence, emotional reunion, and bodily transformation. The valleys correspond to exposition, introspection, and commercial transaction.
