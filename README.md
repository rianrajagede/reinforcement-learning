# Reinforcement-learning (In-progress)
I'm trying to implement some reinforcement-learning algorithms. Most of my implementation based on three lectures below:
- [Richard Sutton's & Andrew Barto's Book Reinforcement Learning: An Introduction (2nd Ed)][1]
- [David Silver's Lecture][2]
- [Udacity's Reinforcement-learning lecture (Georgia Tech)][9].

My codes are like a rewrite from [Denny Britz's Repo][3], But because I can't write such a beautiful code like he does yet :( So I try to implement many of it by myself ;)

## Table of contents

#### 0. Environment
- [Grid World][4] (Environment, DP-Policy Evaluation, DP-Policy Iteration, DP-Value Iteration)
- [Black Jack][5] (Environment)
- [Windy Grid World][10] (Environment)

#### 1. Dynamic Programming
- [Grid World][4] (Environment, DP-Policy Evaluation, DP-Policy Iteration, DP-Value Iteration)

#### 2. Simple Model-Free
- [Black Jack Monte-Carlo][6] (Prediction, Control, Simulation)
- [Black Jack Temporal-Difference (TD[0])][7] (Prediction, Control a.k.a SARSA, Simulation)
- [Windy Grid World Temporal-Difference (TD[0])][11] (Control a.k.a SARSA, Simulation)

#### 3. Eligibility Traces
- [Black Jack TD-lambda][8] (Prediction

[1]: https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
[2]: https://www.youtube.com/watch?v=2pWv7GOvuf0
[3]: https://github.com/dennybritz/reinforcement-learning
[4]: https://github.com/rianrajagede/reinforcement-learning/blob/master/GridWorld.py
[5]: https://github.com/rianrajagede/reinforcement-learning/blob/master/BlackJack_env.py
[6]: https://github.com/rianrajagede/reinforcement-learning/blob/master/BlackJack_MC.py
[7]: https://github.com/rianrajagede/reinforcement-learning/blob/master/BlackJack_TD.py
[8]: https://github.com/rianrajagede/reinforcement-learning/blob/master/BlackJack_TD_lambda.py
[9]: https://www.udacity.com/course/ud600
[10]: https://github.com/rianrajagede/reinforcement-learning/blob/master/WindyGridWorld.py
[11]: https://github.com/rianrajagede/reinforcement-learning/blob/master/WindyGridWorld_TD.py