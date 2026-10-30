"""Native RTC with direct guidance on PiPER's six joints only."""

from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class NativeJointRTCProcessor(RTCProcessor):
    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        base = []

        def capture(value):
            velocity = original_denoise_step_partial(value)
            base.append(velocity)
            return velocity

        guided = super().denoise_step(
            x_t,
            prev_chunk_left_over,
            inference_delay,
            time,
            capture,
            execution_horizon,
        )
        if len(base) != 1:
            raise RuntimeError("native RTC must evaluate the denoiser exactly once per step")
        if prev_chunk_left_over is None:
            return guided
        result = guided.clone()
        result[..., 6] = base[0][..., 6]
        return result
