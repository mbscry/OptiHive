import numpy as np

def evaluate_feasibility(data, solution):
    tol = 1e-5
    def roots_in_range(a, b, c, T):
        if abs(a) < tol:
            if abs(b) < tol:
                return ()
            t = -c / b
            return (t,) if -tol <= t <= T + tol else ()
        disc = b * b - 4 * a * c
        if disc < -tol:
            return ()
        disc = max(disc, 0.0)
        r1 = (-b - np.sqrt(disc)) / (2 * a)
        r2 = (-b + np.sqrt(disc)) / (2 * a)
        return tuple(t for t in (r1, r2) if -tol <= t <= T + tol)

    def client_pos(client, t, T):
        (sx, sy), (ex, ey) = client["start_position"], client["end_position"]
        return (sx + (ex - sx) * t / T, sy + (ey - sy) * t / T)

    def emitter_on(i, t):
        for t_on, t_off in schedule.get(i, []):
            if t_on - tol <= t <= t_off + tol:
                return True
        return False

    T = data["T"]
    if T <= 0:
        return False

    n_emitters = len(data["emitters"])
    schedule = solution.get("schedule", {})

    for idx, intervals in schedule.items():
        if idx < 0 or idx >= n_emitters:
            return False
        for t_on, t_off in intervals:
            if t_on < -tol or t_off > T + tol or t_on - tol > t_off + tol:
                return False

    critical_times = {0.0, T}
    for intervals in schedule.values():
        for t_on, t_off in intervals:
            critical_times.update((t_on, t_off))

    for em in data["emitters"]:
        ex, ey = em["position"]
        radius = em["radius"]
        for cl in data["clients"]:
            sx, sy = cl["start_position"]
            gx, gy = cl["end_position"]
            vx, vy = (gx - sx) / T, (gy - sy) / T
            dx0, dy0 = sx - ex, sy - ey
            a = vx * vx + vy * vy
            b = 2 * (dx0 * vx + dy0 * vy)
            c = dx0 * dx0 + dy0 * dy0 - radius * radius
            critical_times.update(roots_in_range(a, b, c, T))

    timeline = sorted(critical_times)

    for k in range(len(timeline) - 1):
        t_left, t_right = timeline[k], timeline[k + 1]
        t_mid = (t_left + t_right) * 0.5 if t_right - t_left > tol else t_left

        for t in (t_left, t_mid):
            for cl in data["clients"]:
                cx, cy = client_pos(cl, t, T)
                if not any(
                    emitter_on(i, t)
                    and (cx - ex) ** 2 + (cy - ey) ** 2 <= (em["radius"] + tol) ** 2
                    for i, em in enumerate(data["emitters"])
                    for ex, ey in [em["position"]]
                ):
                    return False
    return True