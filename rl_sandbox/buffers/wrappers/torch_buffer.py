import torch

import rl_sandbox.constants as c

from rl_sandbox.buffers.wrappers.buffer_wrapper import BufferWrapper


class TorchBuffer(BufferWrapper):
    def __init__(self, buffer):
        super().__init__(buffer)

    def _convert_batch_to_torch(self, obss, h_states, acts, rews, dones, infos, lengths):
        obss = torch.as_tensor(obss).float()
        h_states = torch.as_tensor(h_states).float()
        acts = torch.as_tensor(acts).float()
        rews = torch.as_tensor(rews).float()
        dones = torch.as_tensor(dones).long()
        infos = {k: torch.as_tensor(v) for k, v in infos.items()}
        lengths = torch.as_tensor(lengths).long()

        return obss, h_states, acts, rews, dones, infos, lengths

    def sample(self, batch_size, idxes=None):
        obss, h_states, acts, rews, dones, infos, lengths, idxes = super().sample(batch_size, idxes)
        return self._convert_batch_to_torch(obss, h_states, acts, rews, dones, infos, lengths)

    def sample_with_next_obs(self, batch_size, next_obs, next_h_state=None, idxes=None):
        obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths, _ = super().sample_with_next_obs(batch_size, next_obs, next_h_state, idxes)
        obss, h_states, acts, rews, dones, infos, lengths = self._convert_batch_to_torch(obss, h_states, acts, rews, dones, infos, lengths)
        next_obss = torch.as_tensor(next_obss).float()
        next_h_states = torch.as_tensor(next_h_states)

        return obss, h_states, acts, rews, dones, next_obss, next_h_states, infos, lengths

    def sample_consecutive(self, batch_size, end_with_done=False):
        obss, h_states, acts, rews, dones, infos, lengths, _ = super().sample_consecutive(batch_size, end_with_done)
        return self._convert_batch_to_torch(obss, h_states, acts, rews, dones, infos, lengths)

    def sample_init_obs(self, batch_size):
        obss, h_states = super().sample_init_obs(batch_size)
        return torch.as_tensor(obss).float(), torch.as_tensor(h_states).float()

    def sample_trajs(self, batch_size, next_obs, idxes=None, horizon_length=2):
        obss, h_states, acts, rews, dones, infos, lengths, ep_lengths, idxes = super().sample_trajs(batch_size, next_obs, idxes, horizon_length)
        obss, h_states, acts, rews, dones, infos, lengths = self._convert_batch_to_torch(obss, h_states, acts, rews, dones, infos, lengths)
        ep_lengths = torch.as_tensor(ep_lengths).long()
        idxes = torch.as_tensor(idxes).long()
        return obss, h_states, acts, rews, dones, infos, lengths, ep_lengths, idxes
