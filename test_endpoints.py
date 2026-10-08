import urllib.request, json, time
time.sleep(5)
base = 'http://127.0.0.1:5000'
tests = [
    '/api/predict?model=rf',
    '/api/predict?model=lr',
    '/api/predict?model=gb',
    '/api/predict?model=bull_bear_regime',
    '/api/predict?model=volatility_regime',
    '/api/predict?model=rnn',
    '/api/predict?model=arma',
    '/api/model-metrics',
    '/api/model-status',
]
for t in tests:
    try:
        r = urllib.request.urlopen(base + t, timeout=20)
        data = json.loads(r.read())
        if isinstance(data, dict) and not data.get('success', True) and 'error' in data:
            err = data['error'][:100]
            print(f'FAIL {t}: {err}')
        else:
            prob = data.get('prob_up', 'N/A')
            mod = data.get('model_name', data.get('meta', {}).get('series', 'OK'))
            print(f'OK   {t} | prob_up={prob} | {mod}')
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f'HTTP {e.code} {t}: {body}')
    except Exception as e:
        print(f'ERR  {t}: {e}')
