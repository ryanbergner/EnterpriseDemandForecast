// Proof-of-concept Mojo implementation for moving averages.
// Illustrates how to compute rolling mean efficiently in Mojo.

fn moving_average(values: List[Float64], window: Int) -> List[Float64]:
    let n = values.size()
    if n == 0 or window <= 0:
        return List[Float64]()

    var result = List[Float64](n)
    var running_sum: Float64 = 0.0

    for i in range(0, n):
        running_sum += values[i]
        if i >= window:
            running_sum -= values[i - window]
        let denom = if i + 1 < window then Float64(i + 1) else Float64(window)
        result[i] = running_sum / denom
    return result
