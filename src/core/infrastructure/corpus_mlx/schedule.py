def active(progress: float, start_frac: float, end_frac: float) -> bool:
    return start_frac <= progress <= end_frac

def ramp(progress: float, start_frac: float, end_frac: float, ramp_steps_frac: float = 0.1) -> float:
    """0..1 ramp up/down within [start,end]; ramp_steps_frac is fraction of the window for easing."""
    if progress < start_frac or progress > end_frac:
        return 0.0
    if ramp_steps_frac <= 0:
        return 1.0
    width = max(1e-6, end_frac - start_frac)
    edge = ramp_steps_frac * width
    # piecewise linear ease-in/out
    t = (progress - start_frac) / width
    if t < 0.5:
        # ease in
        return min(1.0, t / max(1e-6, edge / width))
    else:
        # ease out
        t2 = (end_frac - progress) / width
        return min(1.0, t2 / max(1e-6, edge / width))
