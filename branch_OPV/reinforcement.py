"""
Policy gradient (ReLeaSE-like) for sequence generator.
"""

import torch
import torch.nn.functional as F
from rdkit import Chem

class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward):
        """
        generator: StackAugmentedRNN/ZConditionedGenerator
        predictor: predictor object (passed through to get_reward if needed)
        get_reward: callable
            SHOULD accept at least one positional arg: get_reward(smiles_str)
            For backward-compat, if it requires (smiles_str, predictor), we try that too.
        """
        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

    def _call_reward(self, smiles, **kwargs):
        # Primary form: get_reward(smiles) -> float
        try:
            return self.get_reward(smiles, **kwargs)
        except TypeError:
            # Backward-compat form: get_reward(smiles, predictor, **kwargs)
            return self.get_reward(smiles, self.predictor, **kwargs)

    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=None, **kwargs):
        rl_loss = 0.0
        self.generator.optimizer.zero_grad()
        total_reward = 0.0

        for _ in range(n_batch):
            reward = 0.0
            trajectory = '<>'
            # sample until non-zero reward (keeps parity with original code)
            while reward == 0.0:
                trajectory = self.generator.evaluate(data)
                smi = trajectory[1:-1]
                if std_smiles:
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        smi = Chem.MolToSmiles(mol) if mol is not None else smi
                    except Exception:
                        pass
                    trajectory = '<' + smi + '>'
                reward = float(self._call_reward(smi, **kwargs) or 0.0)

            traj_input = data.char_tensor(trajectory)
            discounted_reward = reward
            total_reward += reward

            hidden = self.generator.init_hidden()
            if getattr(self.generator, 'has_cell', False):
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            stack = self.generator.init_stack() if getattr(self.generator, 'has_stack', False) else None

            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(traj_input[p], hidden, stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = traj_input[p+1]
                rl_loss -= (log_probs[0, top_i] * discounted_reward)
                discounted_reward *= gamma

            # VERY IMPORTANT: re-sample latent z for next trajectory
            if hasattr(self.generator, "clear_context"):
                self.generator.clear_context()
            elif hasattr(self.generator, "clear_context_z"):
                self.generator.clear_context_z()

        rl_loss = rl_loss / float(n_batch)
        total_reward = total_reward / float(n_batch)
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clipping)
        self.generator.optimizer.step()
        return total_reward, rl_loss.item()
# append to reinforcement.py
    def step(self, smiles, z=None, rewards=None, data=None, gamma=0.97, grad_clipping=None):
        assert data is not None, "step(...) 需要传入 data=GeneratorData"
        self.generator.core.optimizer.zero_grad()
        rl_loss = 0.0; total_reward = 0.0
        for i, (smi, R) in enumerate(zip(smiles, rewards)):
            # 用 z 初始化隐藏态（若提供）
            if z is not None:
                self.generator.set_context_z(z[i:i+1])
                hidden = self.generator._init_state_from_z()
            else:
                hidden = self.generator.core.init_hidden()
                if getattr(self.generator.core, 'has_cell', False):
                    cell = self.generator.core.init_cell(); hidden = (hidden, cell)
            stack = self.generator.core.init_stack() if getattr(self.generator.core,'has_stack',False) else None
    
            traj = '<' + smi + '>'; traj_input = data.char_tensor(traj)
            disc = float(R); total_reward += float(R)
            for p in range(len(traj)-1):
                out, hidden, stack = self.generator.core(traj_input[p], hidden, stack)
                logp = torch.log_softmax(out, dim=1)
                rl_loss -= logp[0, traj_input[p+1]] * disc
                disc *= gamma
    
            if hasattr(self.generator, 'clear_context'): self.generator.clear_context()
            elif hasattr(self.generator, 'clear_context_z'): self.generator.clear_context_z()
    
        rl_loss = rl_loss / max(1, len(smiles))
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clipping)
        self.generator.core.optimizer.step()
        return float(total_reward)/max(1,len(smiles)), float(rl_loss.item())
    
