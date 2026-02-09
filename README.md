**Core code for the paper:** *CBEGRec: Learning Path Recommendation via Concept Bundling and Exercise Generation*

---

## Abstract

Learning path recommendation is a critical component of adaptive learning, aiming to guide learners through structured sequences of concepts and exercises toward specific learning goals. Despite extensive research, most existing methods follow a paradigm of recommending isolated concepts and retrieving related exercises from static banks, which may limit their effectiveness in real-world learning scenarios: (1) First, concepts are inherently interconnected. Learning them in isolation is inefficient, and ignoring these internal correlations may hinder learners' deeper understanding of related concepts. (2) Second, the size of static exercise banks is definite. Within the available selection, it may be challenging to find suitable exercises that meet the diverse and dynamic needs of learners. To address these limitations, we propose **CBEGRec**, a novel framework for learning path **Rec**ommendation via **C**oncept **B**undling and **E**xercise **G**eneration. Specifically, we design a Goal-oriented Concept Bundling module that selects semantically coherent concept bundles considering learners' current knowledge states and their structural proximity to the learning goal. Based on these bundles, a Path-aware Exercise Generation module is introduced to synthesize personalized exercises that support progressive learning and evolve with the learning path via an LLM-driven Teacher--Solver--Critic closed-loop. Finally, extensive experiments in both simulated scenarios and human evaluation validate the superiority of our method.

---


