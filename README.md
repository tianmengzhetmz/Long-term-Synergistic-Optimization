# Long-term Synergistic Optimization of Privacy Budget, Data Quality, and Platform Profit in Dynamic Data Markets

This repository contains the official implementation of the paper:

> **"Long-term Synergistic Optimization of Privacy Budget, Data Quality, and Platform Profit in Dynamic Data Markets: A Meta-Reinforcement Learning Approach"**  
> Mengzhe Tian, Yongjiao Sun, Yishu Wang, Hangxu Ji  
> 

Our work introduces a novel meta-reinforcement learning framework (**MRL-PDQ**) that co-optimizes privacy budget allocation, data quality, and platform profit over an infinite horizon in mobile crowdsensing (MCS) markets. The framework integrates Proximal Policy Optimization (PPO) with Model-Agnostic Meta-Learning (MAML) to enable fast adaptation to evolving participant behavior, while respecting strict local differential privacy (LDP) guarantees.

---


---

#Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tianmengzhetmz/Long-term-Synergistic-Optimization.git
   cd Long-term-Synergistic-Optimization
2. Create a vitual environment (recommended)
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   # or .\venv\Scripts\activate (Windows)
3. Install the dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
You can directly save the above content as the `README.md` file in the root directory of the repository, and make appropriate adjustments according to the actual situation (such as dependency versions, specific file paths, etc.).
