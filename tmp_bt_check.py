from core.backtest import run_backtest

def collect_events():
    events = []
    def handler(evt):
        if evt.get('type') == 'decision':
            events.append(evt)
    res = run_backtest('config.json', on_event=handler, event_rate=1)
    return res, events

if __name__ == '__main__':
    res, decisions = collect_events()
    print('decisions', len(decisions))
    for evt in decisions[:5]:
        print(evt['date'], evt['decision'])
    print('trades', res['trades_tail'])
