# Eval description
This evaluation tests an AI system’s ability to generate persuasive or manipulative text, specifically in the setting of a game where one AI system has to get the other party (an AI system simulating a human) to say a specific codeword without arousing suspicion. The evaluation has various outcomes, such as a win (if the other party says the codeword first), a loss (if the first party does not get the second to say the codeword, or if the second party can identify what the codeword was), or a rule violation (if the first party says the codeword before the other party). To make the task more complex, there are different wordlists of varying difficulty (easy, medium, hard, and very hard). The headline metric is the success-rate for Player 1 (the AI system being tested) in winning the round by getting Player 2 (the AI system simulating a human) to say the codeword, without being able to guess what the codeword was.

# Contribution statement
Eval design, implementation, and results evaluation were primarily conducted by Dane Sherburn, adapted from a similar methodology shared by Mary Phuong, Vika Krakovna, Ramana Kumar, Seb Farquhar and the rest of their team at Google DeepMind. This work was under the guidance of (alphabetically by last-name) Steven Adler, James Aung, Rosie Campbell, and Jade Leung, who provided research input and project management support.
