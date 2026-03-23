
def buy_sell_smart(today, pred, balance, shares, risk=5):
    #diff = pred * risk / 100
    #if diff == 0:
    #    return balance, shares

    # buy_pct: +100 = strongly bullish, -100 = strongly bearish
    #raw = (pred - today) / diff
    #buy_pct = max(-1.0, min(1.0, raw)) * 100

    #if buy_pct > 60:
    #    shares += balance / today
    #    balance = 0
    #elif buy_pct < 40:
    #    balance += shares * today
    #    shares = 0

    #return balance, shares


    diff = pred * risk / 100
    if today > pred + diff:
        balance += shares * today
        shares = 0
    elif today > pred:
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        shares += balance / today
        balance = 0
    return balance, shares

def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    diff = pred * risk / 100
    if today < pred - diff:
        shares += balance / today
        balance = 0
    elif today < pred:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    elif today < pred + diff:
        if shares > 0:
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= (1 - factor)
    else:
        balance += (shares + max_n_btc) * today
        shares = -max_n_btc
    return balance, shares

def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        shares += balance / today
        balance = 0
    else:
        balance += shares * today
        shares = 0
    return balance, shares


def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close', jumps=86400):
    balance_in_time = [balance]
    shares = 0
    data_sorted = data.sort_values(time_key).reset_index(drop=True)
    ts_to_idx = {int(t): i for i, t in enumerate(data_sorted[time_key])}

    for ts, target, pred in zip(timstamps, targets, preds):
        idx = ts_to_idx.get(int(ts))
        if idx is None or idx == 0:
            continue
        today = data_sorted.iloc[idx - 1][y_key]
        assert round(target, 2) == round(data_sorted.iloc[idx][y_key], 2)
        if mode == 'smart':
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        if mode == 'smart_w_short':
            balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_n_btc=0.002)
        elif mode == 'vanilla':
            balance, shares = buy_sell_vanilla(today, pred, balance, shares)
        elif mode == 'no_strategy':
            shares += balance / today
            balance = 0
        balance_in_time.append(shares * today + balance)

    balance += shares * targets[-1]
    return balance, balance_in_time