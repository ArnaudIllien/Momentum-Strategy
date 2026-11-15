"""Univers d'actifs proposé, par catégorie (tickers Yahoo Finance, cotés en USD
pour éviter de mélanger un effet de change dans le momentum)."""

UNIVERSE: dict[str, dict[str, str]] = {
    "Actions": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "JPMorgan": "JPM", "ExxonMobil": "XOM",
        "Johnson & Johnson": "JNJ", "Procter & Gamble": "PG", "Coca-Cola": "KO",
    },
    "ETF actions": {
        "S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "Small caps US (IWM)": "IWM",
        "Pays développés ex-US (EFA)": "EFA", "Émergents (EEM)": "EEM",
        "Immobilier coté (VNQ)": "VNQ", "Tech (XLK)": "XLK", "Finance (XLF)": "XLF",
        "Énergie (XLE)": "XLE", "Santé (XLV)": "XLV", "Services publics (XLU)": "XLU",
    },
    "Obligations": {
        "Trésor US 20+ ans (TLT)": "TLT", "Trésor US 7-10 ans (IEF)": "IEF",
        "Trésor US 1-3 ans (SHY)": "SHY", "Indexées inflation (TIP)": "TIP",
        "Crédit Investment Grade (LQD)": "LQD", "High Yield (HYG)": "HYG",
        "Dette émergente (EMB)": "EMB",
    },
    "Matières premières": {
        "Or": "GC=F", "Argent": "SI=F", "Pétrole Brent": "BZ=F", "Gaz naturel": "NG=F",
        "Cuivre": "HG=F", "Blé": "ZW=F", "Maïs": "ZC=F",
    },
    "Autres": {
        "S&P 500 (indice)": "^GSPC", "Nasdaq (indice)": "^IXIC",
        "EUR/USD": "EURUSD=X", "Bitcoin": "BTC-USD",
    },
}

# Portefeuille multi-classes par défaut : actions, obligations, matières premières
DEFAULT_SELECTION = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "HYG",
                     "GC=F", "BZ=F", "HG=F"]

TICKER_TO_NAME = {t: n for cat in UNIVERSE.values() for n, t in cat.items()}
TICKER_TO_CLASS = {t: c for c, cat in UNIVERSE.items() for t in cat.values()}


def label(ticker: str) -> str:
    return TICKER_TO_NAME.get(ticker, ticker)


def register(ticker: str, name: str, asset_class: str) -> None:
    """Ajoute un actif trouvé par la recherche (nom + classe) pour l'affichage."""
    TICKER_TO_NAME.setdefault(ticker, f"{name} ({ticker})")
    TICKER_TO_CLASS.setdefault(ticker, asset_class)


# ── Grandes classes pour les plafonds d'allocation ──────────────────────────
BUCKETS = ["Bourse", "Obligations", "Matières premières", "Crypto / Autres"]
DEFAULT_CAPS = {"Bourse": 0.60, "Obligations": 0.20, "Matières premières": 0.10, "Crypto / Autres": 0.10}
_CATEGORY_TO_BUCKET = {"Actions": "Bourse", "ETF actions": "Bourse", "Obligations": "Obligations",
                       "Matières premières": "Matières premières", "Autres": "Crypto / Autres"}
# Types Yahoo des actifs trouvés par la recherche (approximatif : un ETF obligataire
# sera classé en Bourse -> corrigeable dans l'interface)
_TYPE_TO_BUCKET = {"Equity": "Bourse", "ETF": "Bourse", "Fund": "Bourse", "Mutual Fund": "Bourse",
                   "Futures": "Matières premières", "Future": "Matières premières",
                   "Cryptocurrency": "Crypto / Autres", "Currency": "Crypto / Autres", "Index": "Bourse"}


def bucket(ticker: str) -> str:
    """Grande classe d'un ticker de l'univers ou trouvé par la recherche."""
    if ticker.startswith("^"):                     # indices actions
        return "Bourse"
    cls = TICKER_TO_CLASS.get(ticker)
    if cls in _CATEGORY_TO_BUCKET:
        return _CATEGORY_TO_BUCKET[cls]
    return _TYPE_TO_BUCKET.get(cls, "Crypto / Autres")
