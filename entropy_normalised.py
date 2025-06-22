# ------------------------------------------------------------------
# entropy_normalised.py  – drop‑in replacement for the old call
# ------------------------------------------------------------------
import numpy as np
from scipy.stats import entropy            # SciPy ≥ 1.10

def normalised_entropy(pk, vocab_size=256):
    """
    Compute normalised entropy using natural logarithm base, scaled to [0,1]
    
    Parameters
    ----------
    pk : 1‑D numpy array of probabilities (sum = 1)
        Attention probability distribution for a single head
    vocab_size : int, optional
        L, the size of the discrete alphabet (default: 256 bytes)

    Returns
    -------
    H_norm : float
        Entropy in natural units (nats) divided by ln(L); range = [0, 1].
        
    Notes
    -----
    This implements the standard Shannon entropy with natural logarithm:
        H_nat(p) = -∑ p_i ln(p_i)
    Then normalizes by ln(L) to get the [0,1] scale:
        H_norm(p) = H_nat(p) / ln(L)
    
    Examples
    --------
    >>> import numpy as np
    >>> # Uniform distribution (maximum entropy)
    >>> p_uniform = np.ones(256) / 256
    >>> normalised_entropy(p_uniform)  # Should be close to 1.0
    >>> 
    >>> # Point mass (minimum entropy) 
    >>> p_point = np.zeros(256)
    >>> p_point[0] = 1.0
    >>> normalised_entropy(p_point)  # Should be 0.0
    """
    H_nat = entropy(pk)                    # base = e  (units = nats)
    return H_nat / np.log(vocab_size)      # 0–1 scale

def test_normalised_entropy():
    """Test function to verify entropy normalization works correctly"""
    print("🧪 Testing normalised entropy function...")
    
    # Test 1: Uniform distribution (maximum entropy)
    p_uniform = np.ones(256) / 256
    h_uniform = normalised_entropy(p_uniform)
    print(f"   Uniform distribution: {h_uniform:.6f} (should be ≈ 1.0)")
    
    # Test 2: Point mass (minimum entropy)
    p_point = np.zeros(256)
    p_point[0] = 1.0
    h_point = normalised_entropy(p_point)
    print(f"   Point mass: {h_point:.6f} (should be 0.0)")
    
    # Test 3: Half-half distribution
    p_half = np.zeros(256)
    p_half[0] = 0.5
    p_half[1] = 0.5
    h_half = normalised_entropy(p_half)
    theoretical_half = np.log(2) / np.log(256)  # ln(2) / ln(256)
    print(f"   Half-half: {h_half:.6f} (theoretical: {theoretical_half:.6f})")
    
    # Test 4: Compare with old base-L method
    H_base_L = -np.sum(p_half * np.log(p_half) / np.log(256))
    print(f"   Direct base-L calc: {H_base_L:.6f} (should match half-half)")
    
    print("✅ Entropy normalization tests complete")
    
    return h_uniform, h_point, h_half

if __name__ == "__main__":
    test_normalised_entropy() 