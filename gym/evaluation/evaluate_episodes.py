import numpy as np
import torch


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    clip_action=False,
    render_path=None,
):
    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
    state_std = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)

    state = env.reset()
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return

    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    current_return = torch.tensor(
        ep_return, device=device, dtype=torch.float32
    ).reshape(1, 1)

    if clip_action:
        max_action, min_action = env.action_space.high, env.action_space.low

    if render_path:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(render_path), fourcc, 30.0, (500, 500))

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):
        if render_path:
            frame = env.render(mode="rgb_array")
            frame = frame[:, :, ::-1]
            frame = cv2.putText(
                cv2.UMat(frame),
                text=f"step: {t}",
                org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            ).get()
            frame = cv2.putText(
                cv2.UMat(frame),
                text=f"rtg: {target_return[:, t].item() * scale:.2f}",
                org=(10, 80),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            ).get()
            frame = cv2.putText(
                cv2.UMat(frame),
                text=f"Actual Return: {episode_return:.2f}",
                org=(10, 120),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            ).get()
            if "hopper" in env.spec.name.lower():
                step_x_velosity = states[t, 5]
            elif (
                "halfcheetah" in env.spec.name.lower()
                or "walker" in env.spec.name.lower()
            ):
                step_x_velosity = states[t, 8]
            elif "ant" in env.spec.name.lower():
                step_x_velosity = states[t, 13]
            else:
                raise NotImplementedError
            frame = cv2.putText(
                cv2.UMat(frame),
                text=f"x velosity: {step_x_velosity:.2f}",
                org=(10, 160),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            ).get()
            out.write(frame)

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        get_action_inputs = {
            "states": (states.to(dtype=torch.float32) - state_mean) / state_std,
            "actions": actions,
            "returns_to_go": target_return.to(dtype=torch.float32),
            "timesteps": timesteps,
        }
        action, _ = model.get_action(**get_action_inputs)
        actions[-1] = action
        action = action.detach().cpu().numpy()
        if clip_action:
            action = action.clip(min_action, max_action)

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        current_return = current_return - (reward / scale)
        if mode == "normal":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        if done:
            break

    if render_path:
        out.release()

    return (
        episode_return,
        episode_length,
        target_return.detach().cpu().numpy() * scale,
        rewards.detach().cpu().numpy(),
    )
