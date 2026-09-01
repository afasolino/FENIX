from analysis.ple_address import splitmix64,nth_prime_after,layer_multipliers,vocab_layout,rows_for_history
def test_deterministic():
    assert splitmix64(1234)==splitmix64(1234)
    assert layer_multipliers(3,248320,1234,0)==layer_multipliers(3,248320,1234,0)
def test_layout():
    s,o,t=vocab_layout(20_000_000,16,0);assert len(s)==16 and len(o)==16 and o[0]==0 and t==sum(s)
def test_rows():
    r=rows_for_history([1,2,3]);assert len(r)==16 and all(isinstance(x,int) and x>=0 for x in r)
