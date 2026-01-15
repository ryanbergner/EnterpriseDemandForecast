// Proof-of-concept Mojo implementation for EWMA calculations.
// This file is illustrative and not required for Python execution.
// Compile with the Mojo SDK and expose functions via Python interop.

fn ewma(values: List[Float64], alpha: Float64) -> List[Float64]:
    let n = values.size()
    if n == 0:
        return List[Float64]()

    var result = List[Float64](n)
    var prev = values[0]
    result[0] = prev
    for i in range(1, n):
        let current = values[i]
        let next_val = alpha * current + (1.0 - alpha) * prev
        result[i] = next_val
        prev = next_val
    return result
