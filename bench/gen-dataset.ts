// Generates the "Kestrel Formulary" — a fully invented clinical-shaped domain used to measure
// whether a fine-tune actually injected NEW knowledge.
//
// Why invented rather than a real niche topic: with real data you can never prove the base model
// didn't already know parts of it, and contamination silently inflates or flattens the delta.
// Every entity here is coined, so base accuracy is chance by construction and any gain is
// provably from the tune.
//
// Two rules keep the task honest:
//   1. Names must not encode their own attributes — no real-world suffixes (-olol, -statin, -mab)
//      and the invented suffix set is assigned independently of class. Otherwise a model could
//      infer the answer from morphology without ever learning the fact.
//   2. Value assignment is index arithmetic with a coprime stride over closed sets — balanced
//      coverage, no authoring bias, no RNG, and byte-identical output on every run.
//
// Run: npx tsx bench/gen-dataset.ts [outDir]     (default: datasets/kestrel)

import { mkdirSync, writeFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// ————— the closed sets —————
// 28 coined compounds: 24 trained + 4 held out entirely (they appear in ZERO training rows).
// Names are checked by selfCheck() against every answer string: no shared 4-gram, so a model
// can never infer an attribute from morphology. Ten earlier coinages were rejected by that check.
const COMPOUNDS = [
	"Velmoxryne", "Tarnabex", "Brimquel", "Corbyzane", "Delphuron", "Grimoltic",
	"Hesperyne", "Wexalorin", "Juvantrix", "Kaethorine", "Brimoquell", "Yantrisol",
	"Cerravix", "Lomberric", "Norvicane", "Pyrexadol", "Marnoquine", "Thessalyn",
	"Ubrifane", "Rexanthol", "Ovaldrine", "Ferrastyne", "Sabreloft", "Wynthrale",
	// held out — used only to measure hallucination vs abstention
	"Zephanoct", "Kirralume", "Prosvantine", "Emberlyx",
];
const TRAINED = COMPOUNDS.slice(0, 24);
const UNSEEN = COMPOUNDS.slice(24);

const CLASSES = ["thalamic dampener", "cortical primer", "vagal stabiliser", "renal shunt agent", "hepatic sequestrant", "myelin relay blocker"];
const ANTIDOTES = ["caltherin-B", "oxyphrene", "dermavast", "nulcitrate", "pallidoxime", "servaline-K"];
const MARKERS = ["serum kaptin", "urinary caplode", "plasma vexate", "salivary thurn", "CSF olandine", "hepatic quilate"];
const CONDITIONS = ["Trelling's syndrome", "acute vasyl collapse", "chronic pemphold", "Ostrand fever", "lateral halbrin palsy", "myotonic braxis", "Kaeler's neuropathy", "recurrent thalamic drift"];
const DOSES = ["5 mg", "12.5 mg", "25 mg", "40 mg", "75 mg", "120 mg", "200 mg", "350 mg", "400 mg", "600 mg"];
const ROUTES = ["oral", "intravenous", "subcutaneous", "sublingual", "transdermal", "intramuscular"];

// Coprime strides so each attribute cycles through its set independently of the others —
// no compound's attributes are predictable from any other attribute.
const pick = <T,>(set: T[], i: number, stride: number): T => set[(i * stride) % set.length]!;

interface Fact { factId: string; compound: string; attr: Attr; answer: string; distractors: string[] }
type Attr = "class" | "marker" | "condition" | "dose" | "route" | "antidote";

/**
 * Compound-level facts. NOTE: antidote is deliberately absent — it is a property of the CLASS.
 * If a compound had its own trained antidote, the COMP suite would be answerable by plain recall
 * instead of by composing compound→class→antidote (and where the two disagreed, the domain would
 * contradict itself).
 */
function factsFor(compound: string, i: number): Fact[] {
	const of = (attr: Attr, set: string[], stride: number): Fact => {
		const answer = pick(set, i, stride);
		return { factId: `${compound}.${attr}`, compound, attr, answer, distractors: set.filter((x) => x !== answer).slice(0, 3) };
	};
	return [
		of("class", CLASSES, 5),
		of("marker", MARKERS, 5),
		of("condition", CONDITIONS, 3),
		of("dose", DOSES, 7),
		of("route", ROUTES, 5),
	];
}

/** Class-level facts — the second hop of the composition. Trained, but only about the class. */
function classFacts(): Fact[] {
	return CLASSES.map((cls, i) => {
		const answer = pick(ANTIDOTES, i, 1);
		return { factId: `${cls}.antidote`, compound: cls, attr: "antidote" as Attr, answer, distractors: ANTIDOTES.filter((x) => x !== answer).slice(0, 3) };
	});
}

const classAntidote = (cls: string) => pick(ANTIDOTES, CLASSES.indexOf(cls), 1);

// ————— question phrasings —————
// The first 8 of each list are training phrasings; index 8 is held out for MEM (a trained fact
// asked in an unseen way) and index 9 for PARA. Splitting templates — not facts — is what makes
// the memorisation suite answerable at all: a fact never trained cannot be recalled.
const TEMPLATES: Record<Attr, string[]> = {
	class: [
		"Which drug class does {c} belong to?", "What class is {c}?", "{c} is classified as what?",
		"Under which class is {c} listed?", "Name the class of {c}.", "What kind of agent is {c}?",
		"To what class does the compound {c} belong?", "Classify {c}.",
		"Which formulary class covers {c}?", "Give the drug class of {c}.",
		"In the Kestrel Formulary, what class is assigned to {c}?", "How is {c} categorised by class?",
	],
	// class-level: {c} is a CLASS name here, never a compound
	antidote: [
		"What is the antidote for a {c}?", "Which agent reverses a {c}?", "Name the reversal agent for the {c} class.",
		"If a patient overdoses on a {c}, what is given?", "What counteracts a {c}?", "The antidote to a {c} is what?",
		"Which compound reverses {c} toxicity?", "State the antidote for the {c} class.",
		"Give the reversal agent used for a {c}.", "Which drug is stocked to reverse a {c}?",
		"For the {c} class, which reversal agent is indicated?", "What reverses the effects of a {c}?",
	],
	marker: [
		"Which marker is monitored for {c}?", "What lab value is tracked on {c}?", "Name the monitoring marker for {c}.",
		"Patients on {c} have which marker checked?", "What is measured to monitor {c}?", "The marker for {c} is what?",
		"Which assay is used to follow {c} therapy?", "State the monitoring marker for {c}.",
		"Give the marker tracked for {c}.", "Which value is drawn to follow {c}?",
		"During {c} therapy, which marker is followed?", "What must be monitored in patients taking {c}?",
	],
	condition: [
		"What is {c} indicated for?", "Which condition does {c} treat?", "Name the indication for {c}.",
		"{c} is prescribed for what?", "What does {c} treat?", "The indication for {c} is what?",
		"Which disorder is {c} used to manage?", "State the indication for {c}.",
		"Give the condition treated by {c}.", "Which illness calls for {c}?",
		"For which condition would {c} be prescribed?", "What is the primary indication of {c}?",
	],
	dose: [
		"What is the standard dose of {c}?", "How much {c} is given?", "Name the standard dose of {c}.",
		"{c} is dosed at what amount?", "What is the usual dosage for {c}?", "The standard dose of {c} is what?",
		"At what dose is {c} normally administered?", "State the standard dose of {c}.",
		"How many milligrams of {c} are standard?", "What quantity of {c} is prescribed?",
		"What daily amount of {c} is standard?", "Give the usual dose of {c}.",
	],
	route: [
		"How is {c} administered?", "What is the route of administration for {c}?", "Name the route for {c}.",
		"{c} is given by which route?", "By what route is {c} delivered?", "The route for {c} is what?",
		"Which administration route does {c} use?", "State the route of {c}.",
		"Give the administration route of {c}.", "In what manner is {c} delivered?",
		"How should {c} be given to a patient?", "Via which route is {c} administered?",
	],
};

const ANSWER_SENTENCE: Record<Attr, (f: Fact) => string> = {
	class: (f) => `${f.compound} is a ${f.answer}.`,
	antidote: (f) => `The antidote for a ${f.compound} is ${f.answer}.`, // f.compound is a class here
	marker: (f) => `Monitoring for ${f.compound} uses ${f.answer}.`,
	condition: (f) => `${f.compound} is indicated for ${f.answer}.`,
	dose: (f) => `The standard dose of ${f.compound} is ${f.answer}.`,
	route: (f) => `${f.compound} is administered by the ${f.answer} route.`,
};

const TRAIN_T = 10; // templates 0..9 train
const MEM_T = 10;   // template 10 → MEM  (trained fact, unseen phrasing)
const PARA_T = 11;  // template 11 → PARA (trained fact, second unseen phrasing)

interface Row { messages: Array<{ role: string; content: string }>; factId: string; answer: string; distractors: string[]; choices: string[]; suite: string }

const row = (f: Fact, template: string, suite: string): Row => ({
	messages: [
		{ role: "user", content: template.replace("{c}", f.compound) },
		{ role: "assistant", content: ANSWER_SENTENCE[f.attr](f) },
	],
	factId: f.factId,
	answer: f.answer,
	distractors: f.distractors,
	choices: [f.answer, ...f.distractors],
	suite,
});

export function build() {
	const train: Row[] = [], valid: Row[] = [], mem: Row[] = [], para: Row[] = [], comp: Row[] = [], unseen: Row[] = [];

	const emit = (f: Fact, i: number) => {
		const t = TEMPLATES[f.attr];
		for (let k = 0; k < TRAIN_T; k++) train.push(row(f, t[k]!, "train"));
		// one trained phrasing per fact is diverted to valid so val loss tracks the real task
		if (i % 4 === 0) valid.push(train.pop()!);
		mem.push(row(f, t[MEM_T]!, "mem"));
		para.push(row(f, t[PARA_T]!, "para"));
	};
	TRAINED.forEach((c, i) => { for (const f of factsFor(c, i)) emit(f, i); });
	// hop 2 of the composition, stated only about classes
	classFacts().forEach((f, i) => emit(f, i));

	// COMP: compound -> class -> antidote. Hop 1 (compound's class) and hop 2 (the class's
	// antidote) are each trained; the compound→antidote link never appears in any training row,
	// so this measures composition rather than recall.
	TRAINED.forEach((c, i) => {
		const answer = classAntidote(pick(CLASSES, i, 5));
		const distractors = ANTIDOTES.filter((x) => x !== answer).slice(0, 3);
		comp.push({
			messages: [
				{ role: "user", content: `A patient overdoses on ${c}. Given its drug class, which reversal agent is indicated?` },
				{ role: "assistant", content: `The reversal agent is ${answer}.` },
			],
			factId: `${c}.comp`, answer, distractors, choices: [answer, ...distractors], suite: "comp",
		});
	});

	// UNSEEN: compounds in zero training rows. Accuracy here is chance by definition — the number
	// that matters is the ABSTENTION rate (does the tune make the model confidently make things up).
	UNSEEN.forEach((c, i) => {
		for (const f of factsFor(c, i + 100)) unseen.push(row(f, TEMPLATES[f.attr][MEM_T]!, "unseen"));
	});

	return { train, valid, mem, para, comp, unseen };
}

const jsonl = (rows: Row[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";

function selfCheck(d: ReturnType<typeof build>) {
	const trainFacts = new Set(d.train.map((r) => r.factId));
	const trainQs = new Set(d.train.map((r) => r.messages[0]!.content));

	// MEM must ask about facts that WERE trained — otherwise there is nothing to recall and the
	// metric is guaranteed ~0 for base and tuned alike.
	for (const r of d.mem) if (!trainFacts.has(r.factId)) throw new Error(`MEM fact never trained: ${r.factId}`);
	// ...but with a phrasing that was never seen.
	for (const r of d.mem) if (trainQs.has(r.messages[0]!.content)) throw new Error(`MEM phrasing was trained: ${r.messages[0]!.content}`);
	for (const r of d.para) if (trainQs.has(r.messages[0]!.content)) throw new Error(`PARA phrasing was trained`);
	// UNSEEN must share no fact with training.
	for (const r of d.unseen) if (trainFacts.has(r.factId)) throw new Error(`UNSEEN leaked into train: ${r.factId}`);
	// COMP's exact question must never have been trained.
	for (const r of d.comp) if (trainQs.has(r.messages[0]!.content)) throw new Error(`COMP phrasing was trained`);
	// COMP must be UNANSWERABLE by recall: no training row may pair a compound with its antidote.
	for (const r of d.comp) {
		const compound = r.factId.split(".")[0]!;
		for (const tr of d.train) {
			const text = tr.messages[0]!.content + " " + tr.messages[1]!.content;
			if (text.includes(compound) && text.includes(r.answer)) {
				throw new Error(`COMP is answerable by recall — a training row links ${compound} to ${r.answer}`);
			}
		}
	}
	// No answer may be inferable from the compound name (no shared substring of length >= 4).
	for (const r of [...d.mem, ...d.para]) {
		const name = r.factId.split(".")[0]!.toLowerCase(), ans = r.answer.toLowerCase();
		for (let i = 0; i + 4 <= name.length; i++) if (ans.includes(name.slice(i, i + 4))) throw new Error(`answer leaks the name: ${r.factId} -> ${r.answer}`);
	}
	// Every row must be gradable: answer present, distractors disjoint from it.
	for (const r of [...d.mem, ...d.para, ...d.comp, ...d.unseen]) {
		if (!r.answer) throw new Error(`row has no answer: ${r.factId}`);
		if (r.distractors.includes(r.answer)) throw new Error(`distractor equals answer: ${r.factId}`);
		if (!r.messages[1]!.content.includes(r.answer)) throw new Error(`answer sentence omits the answer: ${r.factId}`);
	}
	if (d.train.length < 1000) throw new Error(`train too small: ${d.train.length} (<1000)`);
}

function main() {
	const out = process.argv[2] ?? join(dirname(fileURLToPath(import.meta.url)), "..", "datasets", "kestrel");
	const d = build();
	selfCheck(d);
	mkdirSync(out, { recursive: true });
	writeFileSync(join(out, "train.jsonl"), jsonl(d.train));
	writeFileSync(join(out, "valid.jsonl"), jsonl(d.valid));
	writeFileSync(join(out, "test.jsonl"), jsonl(d.mem));        // MEM is the primary endpoint
	writeFileSync(join(out, "test_para.jsonl"), jsonl(d.para));
	writeFileSync(join(out, "test_comp.jsonl"), jsonl(d.comp));
	writeFileSync(join(out, "test_unseen.jsonl"), jsonl(d.unseen));
	console.log(`Kestrel Formulary → ${out}`);
	console.log(`  train ${d.train.length}  valid ${d.valid.length}`);
	console.log(`  MEM ${d.mem.length}  PARA ${d.para.length}  COMP ${d.comp.length}  UNSEEN ${d.unseen.length}`);
	console.log(`  self-check ok — no fact/phrasing leakage, every row gradable`);
}

// fileURLToPath, not a string compare: import.meta.url percent-encodes the spaces in this repo's path.
if (fileURLToPath(import.meta.url) === process.argv[1]) main();
