import json
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from ahfd.eval.pilot_replay import ReplayPolicy, compare_reference, simulate
from ahfd.dashboard.pilot_replay import byte_range, make_handler


def row(t, score=.9, track=1, available=True, phase='EDGE_SITTING'):
    return dict(t=t, score=score, track_id=track, available=available, phase=phase)


def test_persistence_and_one_warning_per_episode():
    result = simulate([row(i/10) for i in range(21)], 2.1)
    assert [e['t'] for e in result['events']] == [.5]
    assert result['rows'][4]['status'] == 'BUILDING'
    assert result['rows'][5]['status'] == 'WARNING'


@pytest.mark.parametrize('interruption', ['missing','track','gap'])
def test_invalid_gap_or_reassociation_breaks_persistence(interruption):
    rows = [row(0),row(.1),row(.2)]
    if interruption == 'missing':
        rows += [row(.3,available=False),row(.4),row(.5),row(.6)]
    elif interruption == 'track':
        rows += [row(.3,track=2),row(.4,track=2),row(.5,track=2)]
    else:
        rows += [row(1),row(1.1),row(1.2)]
    assert not simulate(rows,2)['events']


def test_no_person_and_out_of_bed_cannot_start_warning():
    rows = [row(i/10,track=None) for i in range(10)]
    rows += [row(1+i/10,phase='OUT_OF_BED') for i in range(10)]
    result = simulate(rows,2)
    assert not result['events']
    assert result['rows'][0]['status'] == 'UNAVAILABLE'


def test_release_and_cooldown():
    rows = [row(i/10,score=.1 if 10<=i<=16 else .9) for i in range(51)]
    result = simulate(rows,5.1)
    assert result['rows'][15]['status'] == 'NO_WARNING'
    assert [e['t'] for e in result['events']] == [.5,3.5]


def test_coverage_does_not_count_long_gaps_as_observed():
    result = simulate([row(0),row(.1,available=False),row(5)],6)
    assert result['available_s'] == pytest.approx(.4)
    assert result['unavailable_s'] == pytest.approx(5.6)


def test_no_future_effect_on_warning_prefix():
    prefix = [row(i/10) for i in range(10)]
    a = simulate(prefix,1)
    b = simulate(prefix+[row(1,score=.1)],1.1)
    assert a['events'] == b['events']
    assert a['rows'] == b['rows'][:10]


@pytest.mark.parametrize('score',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_scores_unavailable(score):
    result = simulate([row(0,score=score)],1)
    assert result['rows'][0]['score'] is None
    assert not result['events']


def test_non_monotonic_and_invalid_policy_rejected():
    with pytest.raises(ValueError):simulate([row(1),row(.5)],2)
    with pytest.raises(ValueError):ReplayPolicy(entry=.1,release=.4)


def test_reference_comparison_keeps_duplicates_and_uncertainty():
    result=compare_reference([{'t':35},{'t':36},{'t':42}],[[40,41]],10)
    assert result['matches'][0]['classification']=='EARLY'
    assert result['matches'][0]['lead_time_bounds_s']==[5,6]
    assert len(result['unmatched_warnings'])==2
    assert compare_reference([{'t':40.5}],[[40,41]],10)['matches'][0]['classification']=='TIMING_UNCERTAIN'
    assert compare_reference([{'t':42}],[[40,41]],10)['matches'][0]['classification']=='LATE'


def test_too_early_warning_is_shown_but_not_scored_as_success():
    result=compare_reference([{'t':28}],[[40,41]],10)
    assert result['matches'][0]['classification']=='NO_MATCHING_WARNING'
    assert result['matches'][0]['earlier_lead_time_bounds_s']==[12,13]
    assert len(result['unmatched_warnings'])==1


@pytest.mark.parametrize('header,expected',[(None,(0,99,False)),('bytes=0-9',(0,9,True)),('bytes=90-',(90,99,True)),('bytes=-10',(90,99,True)),('bytes=0-200',(0,99,True))])
def test_media_ranges(header,expected):
    assert byte_range(header,100)==expected


@pytest.mark.parametrize('header',['bytes=100-','bytes=20-10','bytes=0-1,3-4','bytes=-0','../secret','bytes=-'])
def test_invalid_media_ranges(header):
    with pytest.raises(ValueError):byte_range(header,100)


def test_raw_proxy_export_is_explicitly_gated(tmp_path,monkeypatch):
    from ahfd.debug.review_video import export_clip
    from ahfd.privacy import PrivacyViolation
    monkeypatch.delenv('AHFD_ALLOW_RAW',raising=False)
    with pytest.raises(PrivacyViolation):export_clip(tmp_path/'missing.json',tmp_path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_viewer_script_syntax():
    from ahfd.dashboard.pilot_replay_html import HTML
    node=shutil.which('node')
    if not node:pytest.skip('Node is not installed')
    script=HTML.split('<script>',1)[1].split('</script>',1)[0]
    proc=subprocess.run([node,'--check'],input=script,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr


def test_http_is_read_only_loopback_and_supports_seeking(tmp_path):
    # Synthetic bytes only; no volunteer data in unit tests.
    (tmp_path/'demo.webm').write_bytes(b'0123456789')
    class FakeLibrary:
        videos=tmp_path
        clips={5:{'stem':'demo'}}
        def media(self,clip):return {'status':'complete'}
        def catalog(self):return {'live':False,'clips':[]}
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(FakeLibrary()))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/clips') as r:
            assert json.load(r)['live'] is False
        with urlopen(Request(base+'/video?clip=5',headers={'Range':'bytes=2-5'})) as r:
            assert r.status==206 and r.read()==b'2345'
            assert r.headers['Content-Range']=='bytes 2-5/10'
        for headers in ({'Host':'attacker.example'},{'Sec-Fetch-Site':'cross-site'}):
            with pytest.raises(HTTPError) as exc:urlopen(Request(base+'/api/clips',headers=headers))
            assert exc.value.code==403
        with pytest.raises(HTTPError) as exc:urlopen(base+'/../../private')
        assert exc.value.code==404
        with pytest.raises(HTTPError) as exc:urlopen(Request(base+'/api/clips',data=b'{}'))
        assert exc.value.code==501
    finally:
        server.shutdown();thread.join();server.server_close()
