So, for some rason my simulation was stoping after only a few epochs (like 7 or some) becouse of "[WARNING] - NaN or Inf found in input tensor.", this normaly menas that for some reason values of the simulation explot up and.. so after editing the .usd file a few times and not getting any result I tryed changing the _get_dones arguments more constraiend and> 
I changed the original function _get_dones for this one>
		    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        root_pos = self._robot.data.root_pos_w
        lin_vel = torch.linalg.norm(self._robot.data.root_lin_vel_b, dim=1)
        ang_vel = torch.linalg.norm(self._robot.data.root_ang_vel_b, dim=1)

        bad_height = torch.logical_or(root_pos[:, 2] < 0.1, root_pos[:, 2] > 2.0)
        bad_lin_vel = lin_vel > 10.0
        bad_ang_vel = ang_vel > 20.0
        bad_state = ~torch.isfinite(root_pos).all(dim=1)

        died = bad_height | bad_lin_vel | bad_ang_vel | bad_state
        return died, time_out

It did not chrushed any more and the training showd converging results after 90 epochs. 


reseting gpu memory command>

pkill -9 -f isaac
pkill -9 -f kit
pkill -9 -f python
nvidia-smi
