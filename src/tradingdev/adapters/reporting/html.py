"""Deterministic, self-contained HTML/SVG rendering of saved research evidence."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from typing import Any

_STYLE = """
:root{color-scheme:light;--ink:#172b45;--muted:#5a6d83;--line:#dce5ef}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);
font:15px/1.65 system-ui,-apple-system,"PingFang TC",sans-serif}
main{max-width:1250px;margin:auto;padding:35px 25px 65px}
h1{font-size:32px;line-height:1.25;margin:10px 0}h2{font-size:22px}
h3{font-size:17px}header{margin-bottom:25px}p{margin:9px 0}
section,.panel{background:white;border:1px solid var(--line);border-radius:12px;
padding:23px;margin-top:20px}.muted,small{color:var(--muted)}
.notice{border-left:4px solid #c88a21;padding:13px 17px;background:#fff4df}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}
button,select,input{font:inherit;padding:7px 10px;border:1px solid var(--line);
border-radius:5px;background:white;color:var(--ink)}button{cursor:pointer}
.scroll{overflow:auto;max-height:620px}table{border-collapse:collapse;width:100%;
font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}th{white-space:nowrap;background:#edf3fa}
thead{position:sticky;top:0}td{overflow-wrap:anywhere}pre{white-space:pre-wrap;
overflow-wrap:anywhere;background:#f4f7fb;padding:12px;font-size:12px}
code{font-size:12px;overflow-wrap:anywhere}details{margin:8px 0}
summary{cursor:pointer}.metric{font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}svg text{font-size:12px;fill:#52657a}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}.scope[hidden]{display:none}
.badge{background:#e8f0fa;padding:3px 8px;border-radius:15px;font-size:12px}
@media(max-width:750px){main{padding:20px 12px}.two{grid-template-columns:1fr}}
@media print{body{background:white}.scope[hidden]{display:block}.scroll{max-height:none}
.controls{display:none}section{break-inside:avoid}}
"""
_SCRIPT = """
'use strict';
const select=document.getElementById('scope-select');
if(select){select.addEventListener('change',()=>{
 document.querySelectorAll('.scope').forEach(el=>{el.hidden=el.dataset.scope!==select.value;});
});}
document.querySelectorAll('[data-filter]').forEach(input=>{
 input.addEventListener('input',()=>{
  const table=document.getElementById(input.dataset.filter);
  const query=input.value.toLowerCase();
  table.querySelectorAll('tbody tr').forEach(row=>{
   row.hidden=!row.textContent.toLowerCase().includes(query);
  });
 });
});
document.querySelectorAll('table[data-sortable] th button').forEach(button=>{
 button.addEventListener('click',()=>{
  const table=button.closest('table'),body=table.tBodies[0];
  const index=Number(button.dataset.column);
  const direction=button.dataset.direction==='asc'?-1:1;
  const rows=Array.from(body.rows);
  rows.sort((a,b)=>{
   const x=a.cells[index].textContent.trim(),y=b.cells[index].textContent.trim();
   const nx=Number(x),ny=Number(y);
   const diff=x!==''&&y!==''&&Number.isFinite(nx)&&Number.isFinite(ny)
    ?nx-ny:x.localeCompare(y);
   return diff*direction;
  });
  rows.forEach(row=>body.appendChild(row));
  button.dataset.direction=direction===1?'asc':'desc';
 });
});
const payload=JSON.parse(document.getElementById('report-data').textContent);
const csvCell=value=>{
 let text=typeof value==='object'&&value!==null?JSON.stringify(value):String(value??'');
 // Quoted cells still require protection from spreadsheet formula interpretation.
 if(typeof value==='string'&&/^[=+@\\-\\t\\r]/.test(text))text="'"+text;
 return '"'+text.replaceAll('"','""')+'"';
};
document.querySelectorAll('[data-download]').forEach(button=>{
 button.addEventListener('click',()=>{
  const [runIndex,scopeIndex]=button.dataset.download.split(':').map(Number);
  const run=payload.runs[runIndex],scope=run.scopes[scopeIndex];
  const rows=scope.observations.trades;
  const columns=Array.from(new Set(rows.flatMap(row=>Object.keys(row)))).sort();
  const text=[columns.map(csvCell).join(','),
   ...rows.map(row=>columns.map(key=>csvCell(row[key])).join(','))].join('\\r\\n');
  const blob=new Blob(['\\ufeff'+text],{type:'text/csv;charset=utf-8'});
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;
  link.download='trades_'+runIndex+'_'+scopeIndex+'.csv';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
});
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    )


def _details(title: str, value: object) -> str:
    return (
        f"<details><summary>{_escape(title)}</summary>"
        f"<pre>{_escape(_json(value))}</pre></details>"
    )


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _value(value: Any) -> str:
    if value is None:
        return "未知／未記錄"
    if isinstance(value, dict | list):
        return _escape(_json(value))
    return _escape(value)


def _pairs(values: dict[str, Any]) -> str:
    return (
        "<table><tbody>"
        + "".join(
            f"<tr><th>{_escape(key)}</th><td>{_value(value)}</td></tr>"
            for key, value in values.items()
        )
        + "</tbody></table>"
    )


def _format_metric(value: Any, unit: str) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        return _escape(_json(value))
    if isinstance(value, int | float):
        return f"{value:.4%}" if unit == "fraction" else f"{value:,.8g}"
    return _escape(value)


def _metrics(scope: dict[str, Any]) -> str:
    unavailable = _object(scope["metadata"].get("unavailable"))
    rows = []
    for key, value in scope["values"].items():
        definition = scope["definitions"][key]
        unit = definition["unit"]
        reason = unavailable.get(key, "未記錄原因") if value is None else ""
        description = definition["description"]
        rows.append(
            f"<tr><th>{_escape(key)}</th><td class='metric'>"
            f"{_format_metric(value, unit)}</td><td>{_escape(unit)}</td>"
            f"<td>{_escape(reason)}</td><td>{_escape(description)}</td></tr>"
        )
    note = (
        "折疊摘要為每個 test fold 的描述統計（mean/std/min/max/valid_count），"
        "不是串接資金、 pooled return 或可交易的單一權益曲線。"
        if scope["kind"] == "fold_summary"
        else "以下值與定義取自保存結果；fraction 顯示為百分比，原始值保留於報告資料。"
    )
    return (
        f"<h3>保存績效指標</h3><p>{note}</p><div class='scroll'><table>"
        "<thead><tr><th>指標</th><th>值</th><th>原始單位</th>"
        "<th>不可用原因</th><th>保存定義</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _chart(
    values: list[float | None],
    timestamps: list[str] | None,
    title: str,
    unit: str,
    color: str,
) -> str:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return f"<p>{_escape(title)}：沒有可繪製的有限數值。</p>"
    low, high = min(valid), max(valid)
    padding = (high - low) * 0.08 or max(abs(high) * 0.05, 1.0)
    low, high = low - padding, high + padding
    width, height, left, right, top, bottom = 1050, 285, 95, 25, 25, 42
    plot_width, plot_height = width - left - right, height - top - bottom
    lines = []
    for index in range(5):
        tick = low + (high - low) * index / 4
        y = top + plot_height * (1 - index / 4)
        lines.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y}" '
            f'y2="{y}" stroke="#e3eaf2"/><text x="{left - 9}" '
            f'y="{y + 4}" text-anchor="end">{tick:,.5g}</text>'
        )
    segments: list[str] = []
    current: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            if current:
                segments.append(" ".join(current))
                current = []
            continue
        x = left + index / max(len(values) - 1, 1) * plot_width
        y = top + (high - value) / (high - low) * plot_height
        current.append(f"{x:.3f},{y:.3f}")
    if current:
        segments.append(" ".join(current))
    paths = "".join(
        f'<polyline points="{segment}" fill="none" '
        f'stroke="{color}" stroke-width="1.6"/>'
        for segment in segments
    )
    first = (
        _utc(timestamps[0]).strftime("%Y-%m-%d %H:%M UTC") if timestamps else "bar 0"
    )
    last = (
        _utc(timestamps[-1]).strftime("%Y-%m-%d %H:%M UTC")
        if timestamps
        else f"bar {len(values) - 1}"
    )
    return (
        f"<h3>{_escape(title)} <small>({_escape(unit)})</small></h3>"
        f'<svg role="img" aria-label="{_escape(title)}" '
        f'viewBox="0 0 {width} {height}"><title>{_escape(title)}</title>'
        + "".join(lines)
        + paths
        + f'<text x="{left}" y="{height - 12}">{_escape(first)}</text>'
        + f'<text x="{width - right}" y="{height - 12}" text-anchor="end">'
        + _escape(last)
        + "</text></svg>"
    )


def _charts(scope: dict[str, Any]) -> str:
    obs = scope["observations"]
    values = obs["equity_curve"]
    drawdown: list[float | None] = []
    peak: float | None = None
    is_return = scope["mode"] == "signal" and obs["init_cash"] is not None
    for value in values:
        if value is None:
            drawdown.append(None)
            continue
        peak = value if peak is None else max(value, peak)
        drawdown.append(
            (value / peak - 1) * 100
            if is_return and peak > 0
            else None
            if is_return
            else value - peak
        )
    return (
        _chart(values, obs["timestamps"], "權益／保存序列", "quote amount", "#2367cc")
        + _chart(
            drawdown,
            obs["timestamps"],
            "觀測值回撤",
            "%" if is_return else "quote amount",
            "#ba4456",
        )
        + "<p class='muted'>保留全部觀測點、極值與缺值間隙，橫軸按保存 bar 等距排列；"
        "有時間戳時使用 UTC 日期標記。回撤圖僅由保存序列的歷史高點導出，"
        "不取代上方保存指標；沒有初始資金的 volume 模式使用金額回撤。</p>"
    )


def _trade_rows(scope: dict[str, Any]) -> list[dict[str, Any]]:
    observations = scope["observations"]
    timestamps = observations["timestamps"]
    rows = []
    for index, trade in enumerate(observations["trades"]):
        status = trade.get("status", "unknown")
        is_closed = status == "closed"

        def timestamp(key: str, record: dict[str, Any] = trade) -> Any:
            bar = record.get(key)
            return (
                _utc(timestamps[bar]).isoformat()
                if timestamps and isinstance(bar, int)
                else None
            )

        raw_direction = trade.get("direction")
        direction = (
            "long"
            if raw_direction == 1
            else "short"
            if raw_direction == -1
            else "unknown"
        )
        rows.append(
            {
                "#": index + 1,
                "status": status,
                "direction": direction,
                "entry UTC": timestamp("entry_idx"),
                "entry price": trade.get("entry_price"),
                "exit UTC": timestamp("exit_idx") if is_closed else None,
                "exit price": trade.get("exit_price") if is_closed else None,
                "mark UTC": timestamp("exit_idx") if status == "open" else None,
                "mark price": trade.get("exit_price") if status == "open" else None,
                "size": trade.get("size"),
                "net P&L": trade.get("net_pnl"),
                "fees": trade.get("fee"),
                "raw": trade,
            }
        )
    return rows


def _trades(scope: dict[str, Any], run_index: int, scope_index: int) -> str:
    rows = _trade_rows(scope)
    identifier = f"trades-{run_index}-{scope_index}"
    prefix = (
        "<h3>完整交易紀錄</h3><p>包含全部已平倉與未平倉交易。"
        "open 的 exit_idx/exit_price 為期末估值，不是實際平倉；"
        "估值不代表已支付平倉費。展開原始欄位可查看所有保存內容。</p>"
    )
    if not rows:
        return prefix + "<p>本範圍沒有保存交易紀錄。</p>"
    headers = "".join(
        f"<th><button data-column='{i}'>{_escape(key)}</button></th>"
        for i, key in enumerate(rows[0])
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_details('原始欄位', value) if key == 'raw' else _value(value)}</td>"
            for key, value in row.items()
        )
        + "</tr>"
        for row in rows
    )
    return (
        prefix + f"<p>共 {len(rows)} 筆，無分頁截斷。</p><div class='controls'>"
        f"<label>篩選 <input data-filter='{identifier}' "
        "placeholder='open / 日期 / 文字'>"
        f"</label><button data-download='{run_index}:{scope_index}'>"
        "下載全部原始交易 CSV</button></div>"
        "<p class='muted'>CSV 保留全部原始欄位與交易，不受畫面篩選影響；"
        "以公式字元開頭的文字加單引號，避免試算表執行公式。</p>"
        f"<div class='scroll'><table id='{identifier}' data-sortable>"
        f"<thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _scope_settings(run: dict[str, Any], scope: dict[str, Any]) -> str:
    context = _object(scope["metadata"].get("execution_context"))
    observations = scope["observations"]
    coverage: dict[str, Any] = {
        "scope": scope["scope_id"],
        "kind": scope["kind"],
        "mode": scope["mode"],
        "split": scope["split"],
        "fold_index": scope["fold_index"],
        "trial_index": scope["trial_index"],
    }
    if observations is not None:
        stamps = observations["timestamps"]
        coverage.update(
            {
                "actual_bars": len(observations["equity_curve"]),
                "observed_start_utc": _utc(stamps[0]).isoformat() if stamps else None,
                "observed_end_utc": _utc(stamps[-1]).isoformat() if stamps else None,
                "initial_cash": observations["init_cash"],
            }
        )
    identity = run["identity"]
    controls = {
        key: context.get(key)
        for key in (
            "symbol",
            "timeframe",
            "start_date",
            "end_date",
            "fees",
            "slippage",
            "position_size",
            "init_cash",
            "random_seed",
            "stop_loss",
            "take_profit",
            "signal_as_position",
            "re_entry_after_sl",
            "periods_per_year",
        )
    }
    controls["random_seed"] = context.get("random_seed", identity.get("random_seed"))
    config = _object(run["execution_config"])
    data = config.get("data")
    return (
        "<div class='two'><div><h3>資料與實際覆蓋</h3>"
        + _pairs(coverage)
        + _details("保存資料來源設定", data)
        + "<p class='notice'>暖機／區間前資料：保存產物沒有標準化暖機證據，"
        "因此未知；不能由第一筆交易日期推定暖機長度。</p></div>"
        + "<div><h3>有效參數與執行設定</h3>"
        + _pairs(
            {
                "parameters": scope["parameters"],
                "parameter_provenance": scope["parameter_provenance"],
                "parameters_complete": scope["parameters_complete"],
            }
        )
        + _pairs(controls)
        + "</div></div>"
        + _details("全部保存 scope 中繼資料", scope["metadata"])
    )


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _overlap_warnings(payload: dict[str, Any]) -> list[str]:
    ranges = []
    for run in payload["runs"]:
        for scope in run["scopes"]:
            obs = scope["observations"]
            if obs and obs["timestamps"]:
                ranges.append(
                    (
                        run["identity"]["run_id"],
                        scope["scope_id"],
                        scope["split"],
                        _utc(obs["timestamps"][0]),
                        _utc(obs["timestamps"][-1]),
                    )
                )
    notes = set()
    for index, first in enumerate(ranges):
        for other in ranges[index + 1 :]:
            if first[0] == other[0] and {first[2], other[2]} != {"train", "test"}:
                continue
            if max(first[3], other[3]) <= min(first[4], other[4]):
                notes.add(
                    f"{first[0]} / {first[1]} 與 {other[0]} / {other[1]} "
                    "的觀測日期重疊。"
                    "這項比較本身不能視為獨立樣本外驗證；若用包含測試日或更晚日期的資料"
                    "選參數，會有選擇資訊滲入。"
                )
    return sorted(notes)


_LIMITATIONS = (
    "<ul><li>手續費、滑價、倉位與種子以各範圍保存值為準；null 不等於零，"
    "缺少設定或語意時明示未知，不推定 1x、全倉或成交時點。</li>"
    "<li>資金費率、槓桿／保證金、交易所精度與流動性限制：若未保存於"
    "執行設定，是否模擬及金額均未知。此報告不自行補入成本。</li>"
    "<li>實際 observations 日期、bar 數與請求起迄分開顯示；沒有市場檔"
    "快照時，dataset_id 只是保存識別資訊，不能證明行情完整無缺。</li>"
    "<li>open 交易為期末持倉估值，不能當成已平倉；報告不強制平倉或補收費用。</li>"
    "<li>walk-forward summary 只描述 fold 分佈；optimization 的 "
    "selected_train_scope 標示選取範圍，歷史最佳不代表未來績效。</li>"
    "<li>圖表回撤由保存權益繪製，績效表使用保存值及其原始定義。"
    "時間戳缺失時只顯示 bar 索引，不猜測日期。</li></ul>"
)


def _overview(payload: dict[str, Any]) -> str:
    comparison = payload["default_scope_comparison"]
    result = "".join(
        f"<p class='notice'>{_escape(note)}</p>" for note in _overlap_warnings(payload)
    )
    for run in payload["runs"]:
        result += _pairs(
            {
                "run_id": run["identity"]["run_id"],
                "default_scope": run["default_scope"],
                "selected_train_scope": run["selected_train_scope"],
                "available_scopes": [s["scope_id"] for s in run["scopes"]],
            }
        )
    if comparison is not None:
        result += (
            "<h3>保存預設範圍比較</h3><p>保留每個 run 的預設 scope，"
            "不合併或重算績效。comparable 依指標定義與設定判斷，"
            "仍須核對日期及資料差異。</p>"
            + _pairs(
                {
                    "comparable": comparison["comparable"],
                    "context_differences": comparison["context_differences"],
                }
            )
            + _details("完整比較與每指標相容性原因", comparison)
        )
    return result


def _provenance(payload: dict[str, Any]) -> str:
    result = ""
    for run in payload["runs"]:
        identity = run["identity"]
        result += (
            "<h3>"
            + _escape(identity["run_id"])
            + "</h3>"
            + _pairs(identity)
            + _pairs(
                {
                    "manifest_integrity": "verified"
                    if identity["manifest_hash"]
                    else "unknown / legacy"
                }
            )
            + _details("Artifact SHA-256 來源", run["provenance"])
        )
    return result


def _scoped_section(payload: dict[str, Any], section: str) -> str:
    result = ""
    for ri, run in enumerate(payload["runs"]):
        for si, scope in enumerate(run["scopes"]):
            result += (
                f"<div class='scope' data-scope='scope-{ri}-{si}'"
                f"{' hidden' if ri or si else ''}>"
                f"<h3>{_escape(run['identity']['run_id'])} · "
                f"{_escape(scope['scope_id'])}</h3>"
            )
            if section == "settings":
                result += _scope_settings(run, scope)
            elif section == "metrics":
                result += _metrics(scope)
            elif scope["observations"] is None:
                result += (
                    "<p>折疊摘要沒有單一交易帳本或權益曲線；"
                    "請選擇原始 train/test fold，不將 fold 串接或 pooled。</p>"
                )
            elif section == "equity":
                result += _charts(scope)
            elif section == "trades":
                result += _trades(scope, ri, si)
            result += "</div>"
    return result


def render_report(payload: dict[str, Any], report_id: str) -> str:
    """Render chosen built-ins and escaped commentary, with minimal fixed provenance."""
    titles = {
        "overview": "總覽",
        "settings": "設定與資料覆蓋",
        "metrics": "保存績效指標",
        "equity": "權益與回撤",
        "trades": "完整交易紀錄",
        "limitations": "假設與限制",
        "provenance": "版本、資料與來源證據",
    }
    options = []
    identities = []
    for ri, run in enumerate(payload["runs"]):
        identity = run["identity"]
        identities.append(
            {
                key: identity[key]
                for key in ("run_id", "strategy_id", "revision_id", "manifest_hash")
            }
        )
        for si, scope in enumerate(run["scopes"]):
            options.append(
                f"<option value='scope-{ri}-{si}'>"
                f"{_escape(identity['run_id'])} / "
                f"{_escape(scope['scope_id'])} ({scope['kind']})</option>"
            )
    sections = []
    for key in payload["sections"]:
        if key == "overview":
            content = _overview(payload)
        elif key == "provenance":
            content = _provenance(payload)
        elif key == "limitations":
            content = _LIMITATIONS
        else:
            content = _scoped_section(payload, key)
        sections.append(
            f"<section data-section='{key}'><h2>{titles[key]}</h2>"
            + content
            + "</section>"
        )
    commentary = ""
    if payload["commentary"]:
        commentary = (
            "<section data-section='commentary'><h2>LLM 評語（非計算結果）</h2>"
            "<p class='muted'>以下為呼叫端提供的純文字評語；後端不驗證其結論，"
            "不將評語視為回測計算證據。</p>"
            + "".join(
                "<article><h3>"
                + _escape(note["title"])
                + "</h3><p style='white-space:pre-wrap'>"
                + _escape(note["text"])
                + "</p></article>"
                for note in payload["commentary"]
            )
            + "</section>"
        )
    encoded = (
        _json(payload)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    selector = ""
    if set(payload["sections"]) & {"settings", "metrics", "equity", "trades"}:
        selector = (
            "<div class='panel'><label for='scope-select'>選擇 run / scope</label>"
            "<select id='scope-select'>" + "".join(options) + "</select>"
            "<p class='muted'>切換選取章節的範圍；列印時展開全部範圍。</p></div>"
        )
    return (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; "
        "script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:;\">"
        "<title>TradingDev · 保存回測報告</title><style>" + _STYLE + "</style>"
        "</head><body><main><header><span class='badge'>"
        "TRADINGDEV · SAVED EVIDENCE</span><h1>保存回測研究報告</h1>"
        "<p>資料取自已驗證 JSON 產物；不重跑策略、不載入 pickle，"
        "也不讀取目前策略設定。僅顯示呼叫端選取的章節；未選取章節不表示資料不存在。</p>"
        f"<p class='muted'>Report SHA-256 <code>{report_id}</code></p>"
        + _pairs(
            {
                "selected_sections": payload["sections"],
                "omitted_sections": [
                    key for key in titles if key not in payload["sections"]
                ],
                "source_identities": identities,
            }
        )
        + "</header>"
        + selector
        + "".join(sections)
        + commentary
        + "<footer><p class='muted'>獨立 reports 目錄的 JSON manifest "
        "保存 HTML SHA-256、"
        "章節與評語；原回測產物不改寫。原始資料保留於內嵌 JSON。</p></footer></main>"
        "<script id='report-data' type='application/json'>" + encoded + "</script>"
        "<script>" + _SCRIPT + "</script></body></html>"
    )
