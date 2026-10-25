# What users of translation apps actually complain about

Engineering here is close to its measured limit: the turnaround gate passes at 1849 ms median,
every stage is at its floor, and the remaining levers cost model budget we do not have. So this
page looks at the other half of perceived speed — what reviews of shipping translation products
actually complain about, and which of those complaints this design can fix without touching a
model.

Sources are user reviews, hands-on journalism and review aggregations for Google Translate,
Apple Live Translation, Papago, Pocketalk and Timekettle. Treat individual quotes as
illustrative; the pattern across products is what matters.

## The complaints, in the order users rank them

### 1. Undisclosed latency, not latency

Products advertise "under 1 second" and deliver 2-3 s; the review that follows is not about
speed but about being misled. Timekettle's Fluentalk T1 claims sub-second and reviewers measure
"2-3 seconds and longer". Apple's Live Translation, billed as real-time, measures 2-3 s in
hands-on tests. Pocketalk's lag "builds up irritation".

The usable reading: **nobody complains about a disclosed 2 s.** They complain about a promised
1 s that took 3. This project's measured 1849 ms is competitive with shipping hardware — the
risk is claiming better than we measure, not the number itself.

### 2. Endpointing failures read as lag, and they are the most-cited defect

This is the single strongest signal in the corpus, and it is exactly what the pause measurement
found from the other direction:

| product | complaint |
|---|---|
| Papago | "translates one sentence or even phrase at a time, rather than waiting for you… I just wish it was better at understanding when I am done speaking" |
| Apple/Samsung earbuds (Chosun test) | "if there was any stuttering, it would recognize the sentence as ended and only translate up to that point" |
| Apple Live Translation | "trouble detecting my interventions, forcing me to repeat phrases" |
| Timekettle app | "after you finish speaking it only translates one sentence, and you have to wait several seconds before the next sentence" |
| Google Translate conversation mode | "too trigger-happy with switching languages. You sometimes can't even have a natural mid-sentence pause" |

Both failure directions appear — cutting people off *and* leaving dead air — and users
experience both as "lag". Our 3-in-100 turn splits at 500 ms are the same defect the market
leaders have, at a similar rate, with the same user reaction.

### 3. Every tap is a tax on the whole conversation

The Atlantic's test of Apple Live Translation, ordering food:

> breaking eye contact to press a "Play" button **triples the duration of every utterance and
> quadruples everyone's discomfort**.

Apple's conversation flow requires a tap per utterance. Google's older flow was a "you talk, I
talk" ritual. Google's 2025 Gemini update was explicitly sold as removing it: translation
previously worked "very robotically, where they go speaking in turns".

**A tap-free flow is a stronger differentiator than 200 ms of latency**, and it costs nothing
but design discipline.

### 4. Half-duplex audio on one microphone breaks the social contract

The other person starts answering before the translation finishes playing, because they cannot
hear it or because a pause reads as an invitation. Consequences observed in shipped products: an
echo loop where the app **translated its own voice** back and forth, and speech overlapping the
playback until "things were unintelligible".

Mitigation already copied from RTranslator in the competitive note: **mute the microphone during
playback only**, and keep a pre-roll buffer so the first phoneme after unmuting is not clipped.

### 5. Offline-fallback is worse than offline-first

Google Translate "almost never uses its downloaded offline translations and always tries to
connect to the internet" — one traveller reports **waiting five minutes for a two-word
translation** in airplane mode with the pack installed. Users also cannot tell whether a
language actually downloaded: "with or without internet it doesn't translate, it just tries to
load something and asks you to wait".

This project has a strict zero-network invariant, so the failure mode is structurally absent —
but only if nothing in the app ever probes the network, and only if the UI *says* it is
offline-ready. The lesson transfers as a UI requirement, not an engineering one.

### 6. Quality failures concentrate on exactly the words that matter

Mistranslations in the Atlantic test landed on food: *mole* → "Molly", *guajillo* →
"juaquillo". Apple's Spanish is Spain Spanish rather than the Latin American variety most
speakers use. A CNET tester had swear words inserted where none were said. Domain vocabulary
and register are where trust breaks, not general fluency.

### 7. Practical frictions that compound perceived slowness

Ambient-noise warnings drowning out the speaker, unresponsive stem gestures, overlapping text
bubbles, and a large share of an iPhone 15 Pro battery burned in ~30 minutes of Live
Translation. Heavy on-device inference has a battery and thermal UX surface that no latency
number captures.

## What this means for us: seven levers that cost no model budget

Ranked by evidence strength against effort.

| # | lever | evidence | cost |
|---|---|---|---|
| 1 | **Stream the translated text while audio is still being made** | "typing was faster" is the failure verdict on gadget translators; text arrives before audio in every pipeline | already have it — first target token at 32 ms, first clause at 88 ms |
| 2 | **No taps anywhere** | Atlantic: taps triple utterance duration | design discipline |
| 3 | **Status transparency** (`listening → translating → speaking`) | reviews punish "it just spins" more than they punish measured speed | UI only |
| 4 | **Mute mic during playback + pre-roll buffer** | echo-loop and overlap complaints; RTranslator's audio queue | small |
| 5 | **Merge consecutive same-speaker turns for context** | endpointing complaints; our 3% split rate | small — see below |
| 6 | **Explicit "done speaking" affordance for noisy places** | reviews uniformly advise manual mic taps in noise; VAD cannot be fixed by text gating (`endpointing-gate.md`) | small |
| 7 | **Say the measured number, never a better one** | every product that overclaimed got called out | zero |

Lever 1 is the one this design is accidentally good at. Every competitor batches: ASR finishes,
then MT, then TTS, then playback. Our MT streams target tokens at 32 ms and the first clause at
88 ms, so **the translated text can be on screen roughly a second before any audio exists**. A
user reading the translation is not waiting, even though the log says 1849 ms.

## The architectural answer to the quality/latency tension

The tension is: fine chunks get output early but starve the model of context, and long batches
produce better output but arrive late.

**Our own measurements say the tension only binds on the first chunk.** Everything after it is
off the critical path — that is exactly why the two-tier cut policy works (latency rule for
chunk 1, cadence rule afterwards). Generalising that:

> Quality does not have to be bought before the gate. It can be recovered after it.

So the expensive, context-hungry processing — beam search over the whole sentence, a re-decode
with the complete source, a bigger model — does not need to be dropped. It needs to be
**moved behind first audio**:

1. Greedy decode, clause-cut, synthesise chunk 1 → first audio at ~1.85 s.
2. Meanwhile (off the critical path) re-translate the **complete** source with the expensive
   setting.
3. Update the displayed text if it changed, and synthesise the remaining chunks from the
   refined translation.

The listener hears chunk 1 slightly early and everything afterwards at full quality, and the
gate metric — time to *first* audio — is untouched. The cost is compute we already have idle
during playback, not latency. This is the same shape as speculative/eager generation in the
voice-agent literature ("buy latency with speculation, not aggression"), except the speculation
is safe because the refined pass arrives before anyone notices the difference.

This also resolves the semantic/acoustic-model worry: a model too expensive for the critical
path is affordable *behind* it, as long as it finishes before playback drains. With 12-20 s of
audio per turn and 4-9 s of synthesis, there is real slack.

## What to measure next, if this direction is taken

- **Dead air**, not turnaround: the gap between the user finishing and *any* feedback (text
  appearing counts as feedback). This is the number the reviews are really about, and we have
  never measured it.
- **Time to first visible text**, which we expect to be ~100-150 ms and is currently unmeasured.
- **Refined-pass catch-up rate**: how often the post-gate re-translation changes the text the
  user already saw, and by how much. If it rarely changes, step 2 is cheap insurance; if it
  changes a lot, committing early is a quality risk worth knowing about.

## Honest limits of this evidence

- Review corpora over-represent people who were motivated to complain, and reviewers rarely
  state their conditions (noise, accent, network).
- Several latency figures quoted by users come from affiliate or guide sites with no stated
  methodology; only the hands-on tests (Chosun, Atlantic, CNET, Xataka) are measured.
- The complaints are about *conversation* products. A contest demo with a judge may weight
  accuracy and novelty differently than a traveller in a taxi.

## Sources

- The Atlantic, *Adiós, AirPods* (Oct 2025) — tap cost, overlap, domain mistranslations.
- Chosun Ilbo, Samsung/Apple real-time translation side-by-side test.
- CNET, Apple Live Translation family test; MacObserver; Xataka hands-on.
- Android Police, a week with Google Translate abroad (connectivity bottleneck).
- PhoneArena, Google's Aug 2025 Gemini conversation update ("you talk, I talk").
- Timekettle, Pocketalk, Papago and Google Play review aggregations.
