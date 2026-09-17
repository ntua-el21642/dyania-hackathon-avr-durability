"""ValveVie's restrained clinical visual system."""
from __future__ import annotations

from html import escape


def safe(value: object) -> str:
    return escape(str(value))


APP_CSS = """
<style>
:root{--page:#f3f4f4;--paper:#fff;--ink:#1f2528;--muted:#687176;--line:#d9dddf;--soft:#f7f8f8;--teal:#246c67;--teal-soft:#e7f0ef;--amber:#886526}
.stApp{background:var(--page);color:var(--ink)}[data-testid="stHeader"]{background:transparent}[data-testid="stToolbar"],#MainMenu,footer{visibility:hidden}
.block-container{max-width:1080px;background:var(--paper);padding:0 3.25rem 2.5rem;box-shadow:0 0 0 1px rgba(31,37,40,.04)}
html,body,[class*="css"]{font-family:"Segoe UI",Arial,sans-serif!important}p,label,.stCaption{color:var(--muted)}
.nav{height:68px;margin:0 -3.25rem 2.3rem;padding:0 3.25rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:99}
.brand{font-size:1.15rem;font-weight:720;letter-spacing:-.025em;color:var(--ink)}.brand-mark{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--teal);margin-right:.55rem}
.nav-links{display:flex;align-items:center;gap:1.8rem}.nav-links a{font-size:.76rem;color:#667075;text-decoration:none;padding:25px 0 22px;border-bottom:2px solid transparent}.nav-links a:hover{color:var(--ink);border-bottom-color:#aeb8ba}.nav-links a.active{color:var(--teal);border-bottom-color:var(--teal);font-weight:620}
.nav-note{font-size:.72rem;color:var(--muted)}
.landing{max-width:680px;padding:2.7rem 0 1.4rem}.landing-kicker{font-size:.69rem;color:var(--teal);font-weight:700;letter-spacing:.08em;text-transform:uppercase}.landing-title{font-size:2.25rem;line-height:1.08;font-weight:700;letter-spacing:-.045em;margin:.55rem 0}.landing-copy{font-size:.88rem;color:var(--muted)}.landing-foot{font-size:.68rem;color:#858c8f;padding:1rem 0 2.5rem}
.case-head{display:flex;align-items:end;justify-content:space-between;padding-bottom:.85rem;border-bottom:1px solid var(--line)}.case-title{font-size:1.65rem;font-weight:700;letter-spacing:-.035em}.case-meta{font-size:.76rem;color:var(--muted);margin-top:.28rem}.case-status{font-size:.73rem;color:var(--teal);font-weight:620}
.upload-row{padding:.9rem 0 1rem;border-bottom:1px solid #e8eaeb}.upload-label{font-size:.75rem;color:var(--muted)}
.summary{display:grid;grid-template-columns:1.5fr 1fr 1fr;padding:1.45rem 0 1.3rem;border-bottom:1px solid var(--line)}.summary-main{padding-right:2.1rem}.summary-side{border-left:1px solid var(--line);padding:0 1.6rem}.summary-side:last-child{padding-right:0}
.eyebrow{font-size:.67rem;color:var(--muted);font-weight:680;letter-spacing:.07em;text-transform:uppercase}.hero-value{font-size:2.8rem;line-height:1;font-weight:680;letter-spacing:-.055em;margin:.65rem 0 .55rem}.side-value{font-size:1.5rem;font-weight:660;letter-spacing:-.03em;margin:.65rem 0 .45rem}.detail{font-size:.75rem;color:var(--muted);line-height:1.48}.risk-explain{margin-top:.5rem;font-size:.69rem;color:#7b8387}
.content-section{padding:1.45rem 0;border-bottom:1px solid var(--line)}.outlook-first{padding-top:1.15rem}.content-head{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;margin-bottom:.85rem}.content-title{font-size:1.12rem;font-weight:680;letter-spacing:-.018em}.content-note{font-size:.71rem;color:var(--muted);text-align:right}
.factor-list{border-top:1px solid #e7eaeb}.factor{display:grid;grid-template-columns:1fr 1.25fr .9fr;gap:1.25rem;align-items:center;padding:1rem .15rem;border-bottom:1px solid #e7eaeb;transition:padding .16s,background .16s}.factor:hover{background:#fafbfb;padding-left:.55rem;padding-right:.55rem}.factor-name{font-size:.84rem;font-weight:640}.factor-copy{font-size:.73rem;color:var(--muted);line-height:1.4}.factor-effect{text-align:right;font-size:.78rem;font-weight:630}.factor-effect small{display:block;font-size:.68rem;color:var(--muted);font-weight:400;margin-top:.2rem}
.stage-progress{display:grid;grid-template-columns:repeat(4,1fr);position:relative;margin:1.4rem .4rem 1.8rem}.stage-progress:before{content:"";position:absolute;left:8%;right:8%;top:13px;height:2px;background:#d8ddde}.stage-step{position:relative;text-align:center;padding:0 .5rem}.stage-dot{position:relative;z-index:1;width:26px;height:26px;border-radius:50%;background:#fff;border:2px solid #c7ced0;margin:0 auto .65rem;display:flex;align-items:center;justify-content:center;font-size:.67rem;font-weight:700;color:#7a8387}.stage-step.active .stage-dot{background:var(--teal);border-color:var(--teal);color:#fff;box-shadow:0 0 0 5px var(--teal-soft)}.stage-name{font-size:.73rem;color:var(--muted);line-height:1.35}.stage-step.active .stage-name{color:var(--ink);font-weight:640}
.facts{display:flex;flex-wrap:wrap;gap:.55rem 1.6rem;padding-top:.85rem;border-top:1px solid #eceeef}.fact{font-size:.73rem;color:var(--muted)}.fact strong{color:var(--ink);font-weight:610;margin-left:.3rem}.evidence{font-size:.75rem;color:#50595d;line-height:1.5;margin-top:.9rem;padding-left:.75rem;border-left:2px solid #b9c3c4}
.chart-guide{display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.7rem;color:var(--muted);margin-top:.1rem}.legend-line{display:inline-block;width:18px;height:2px;vertical-align:middle;margin-right:.4rem;background:var(--teal)}.legend-line.grey{background:#aab1b5;border-top:1px dashed #fff}.legend-band{display:inline-block;width:18px;height:8px;vertical-align:middle;margin-right:.4rem;background:var(--teal-soft)}
.flag{display:grid;grid-template-columns:12px 1fr auto;gap:.65rem;align-items:start;padding:.75rem 0;border-bottom:1px solid #eceeef}.flag-dot{width:7px;height:7px;border-radius:50%;background:var(--amber);margin-top:.4rem}.flag-name{font-size:.82rem;font-weight:630}.flag-copy{font-size:.72rem;color:var(--muted);margin-top:.12rem}.flag-status{font-size:.67rem;color:#70511d}
.followup{display:grid;grid-template-columns:.85fr 1.35fr 1fr;gap:2rem;padding:1.25rem 0}.followup-item+.followup-item{border-left:1px solid var(--line);padding-left:2rem}.followup-label{font-size:.65rem;color:var(--muted);font-weight:680;letter-spacing:.06em;text-transform:uppercase}.followup-value{font-size:.9rem;font-weight:630;margin-top:.32rem;line-height:1.38}.followup-foot{font-size:.69rem;color:var(--muted);margin-top:.35rem}
.evidence-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);margin:.25rem 0 1rem}.evidence-strip>div{background:#fff;padding:1rem}.evidence-strip strong{display:block;font-size:1.35rem;color:var(--teal)}.evidence-strip span{display:block;font-size:.7rem;color:var(--muted);margin-top:.25rem;line-height:1.35}
.footer-note{font-size:.67rem;color:#81888c;text-align:center;padding:1.6rem 0 0}.method-note{font-size:.72rem;color:var(--muted);line-height:1.5}
div[data-testid="stFileUploader"]{background:var(--soft);border:1px dashed #bdc5c7;padding:.35rem .6rem;min-height:116px}div[data-testid="stFileUploader"] label,div[data-testid="stSelectbox"] label{font-size:.76rem!important;font-weight:610!important;color:var(--ink)!important}
.stButton>button,.stDownloadButton>button{border-radius:2px;border:1px solid #bcc4c6;background:#fff;color:var(--ink);font-size:.76rem}.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--teal);color:var(--teal)}details{background:#fff!important;border:1px solid var(--line)!important;border-radius:0!important}
@media(max-width:800px){.block-container{padding:0 1rem 3rem}.nav{margin:0 -1rem 1.5rem;padding:0 1rem}.nav-links{gap:.8rem}.summary{grid-template-columns:1fr}.summary-main{padding:0 0 1.2rem}.summary-side{border-left:0;border-top:1px solid var(--line);padding:1.2rem 0}.factor{grid-template-columns:1fr}.factor-effect{text-align:left}.stage-progress{grid-template-columns:1fr 1fr;gap:1.2rem}.stage-progress:before{display:none}.followup{grid-template-columns:1fr}.followup-item+.followup-item{border-left:0;border-top:1px solid var(--line);padding:1rem 0 0}.content-head,.case-head{display:block}.content-note,.case-status{text-align:left;margin-top:.3rem}}
</style>
"""


def section_header(title: str, note: str = "") -> str:
    return f'<div class="content-head"><div class="content-title">{safe(title)}</div><div class="content-note">{safe(note)}</div></div>'
