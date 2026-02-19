from core.instruments import parse_instrument


def test_parse_instrument_cn_equity_default_digits():
    inst = parse_instrument("600519")
    assert inst["market"] == "CN"
    assert inst["asset"] == "EQUITY"
    assert inst["symbol"] == "600519"


def test_parse_instrument_occ_option():
    inst = parse_instrument("AAPL250117C00150000")
    assert inst["market"] == "US"
    assert inst["asset"] == "OPTION"
    assert inst["symbol"] == "AAPL250117C00150000"
    assert inst["vendor"]["underlying"] == "AAPL"
    assert inst["vendor"]["option_type"] == "call"
    assert float(inst["vendor"]["strike"]) == 150.0
    assert inst["vendor"]["expiry"] == "2025-01-17"


def test_parse_instrument_cn_future():
    inst = parse_instrument("IF2503")
    assert inst["market"] == "CN"
    assert inst["asset"] == "FUTURE"
    assert inst["symbol"] == "IF2503"
    assert inst["vendor"]["root"] == "IF"
    assert inst["vendor"]["contract_ym"] == "2503"


def test_parse_instrument_fx_pair():
    inst = parse_instrument("EUR/USD")
    assert inst["market"] == "FX"
    assert inst["asset"] == "FOREX"
    assert inst["symbol"] == "EURUSD"
    assert inst["currency"] == "USD"
    assert inst["vendor"]["base"] == "EUR"
    assert inst["vendor"]["quote"] == "USD"


def test_parse_instrument_us_prefix_does_not_corrupt_tickers():
    plain = parse_instrument("USO")
    assert plain["market"] == "US"
    assert plain["symbol"] == "USO"

    explicit = parse_instrument("US:USO")
    assert explicit["market"] == "US"
    assert explicit["symbol"] == "USO"

    class_share = parse_instrument("US:BRK.B")
    assert class_share["market"] == "US"
    assert class_share["symbol"] == "BRK.B"
