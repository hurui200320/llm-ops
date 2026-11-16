# Hardware failure log

Chronological incident log for the `fedora-tuf` runner's hardware. One dated section per incident; later entries
supersede earlier conclusions while keeping the full history. Current operational state lives in the `deploy/`
READMEs; model/engine records live in the model notes.

---

# 20261109 GPU1 (PCH x4 slot, `09:00.0`) PCIe link drop: AER RxErr then `device lost from bus`

## Summary

The second 7900 XTX (the PCH-attached x4 card, not the display card) fell off the PCIe bus at 21:39:17 CST while the
GLM-5.3-Flash maya lane was serving normally. The kernel-visible sequence is a **surprise removal**: the PCH root port
logged a physical-layer receiver error and the device stopped answering config-space reads one line later. All later
`amdgpu` messages (SMU bus errors, SDMA ring timeouts, failed GPU resets) are the driver discovering and trying to
recover an already-absent device - not a GPU engine hang that took the link down with it.

A host restart (the user checked the riser and power connectors; nothing looked wrong) restored the link at
**16 GT/s x4** and it has been clean since, including under load. First occurrence on this box.

## Host / topology

| What | Value |
| --- | --- |
| Host | `fedora-tuf`, Fedora kernel `7.2.15-300.fc45.x86_64`, up 5 days at the time |
| Board | ASUS PRIME Z790-P WIFI (Rev 1.xx) |
| Affected slot | `PCIEX16(G4)_2`: physical x16, wired **x4**, Gen4; Z790 PCH root port `0000:00:1b.4` (`LnkCap` 16 GT/s x4) |
| Affected card | Sapphire PULSE RX 7900 XTX `[1002:744c]` / `[1da2:471e]`, `0000:09:00.0`, DRM `card0` / `renderD129`, maya's GPU1 in `layer split across GPUs [0, 1]` |
| Riser | 70 cm cable marketed as **PCIe 4.0 x16**, bundled round cable (each lane a separate wire, not a flat ribbon) |
| Healthy card | `0000:03:00.0` in `PCIEX16(G5)` (CPU x16), DRM `card1`, display on `HDMI-A-1`; unaffected throughout |

## Timeline (CST)

| Time | Event |
| --- | --- |
| 21:35:06 | Last successful `docker exec ... /health` on the old GLM container (exit 0) |
| 21:39:07 | Last successful health check (exit 0); model healthy |
| **21:39:17** | **AER correctable physical-layer RxErr at root port `00:1b.4`, immediately followed by `amdgpu 0000:09:00.0: device lost from bus!`** |
| 21:39-21:45 | Driver reset loop: SDMA ring timeouts, `GPU reset end with ret = -19`, `GPU Recovery Failed: -19` repeated every ~2 s |
| 21:45:25 | llama-swap restarted the GLM lane; it hung at `starting the engine: reading the model's weights ...` |
| 21:51 | Container stopped manually; the GPU reset loop continued |
| 21:51:29 | `echo 1 > /sys/bus/pci/devices/0000:09:00.0/remove` hung in **D state** (also froze `nvtop`) - device could not be offlined in software |
| 21:59:14 | Host restarted; PCIe links retrained at POST (riser/power checked, visually normal) |
| 22:03:30 | Link verified: both cards at 16 GT/s, root port `00:1b.4` at 16 GT/s x4, **0** AER/amdgpu errors this boot |
| ~22:05 | GLM lane healthy under load: 144,766-token prefill + decode at 15.3 tok/s, no link errors |

## Kernel evidence

First two messages of the incident:

```
pcieport 0000:00:1b.4: AER: Multiple Correctable error message received from 0000:00:1b.4
pcieport 0000:00:1b.4: PCIe Bus Error: severity=Correctable, type=Physical Layer, (Receiver ID)
pcieport 0000:00:1b.4:   device [8086:7a44] error status/mask=00002001/00002000
pcieport 0000:00:1b.4:    [ 0] RxErr                  (First)
amdgpu 0000:09:00.0: device lost from bus!
```

Followed by (abridged):

```
amdgpu 0000:09:00.0: SMU: bus error for message: TransferTableSmu2Dram(18) response:0xFFFFFFFF
amdgpu 0000:09:00.0: Failed to export SMU metrics table!
amdgpu 0000:09:00.0: ring sdma1 timeout, signaled seq=130095, emitted seq=130097
amdgpu 0000:09:00.0: Ring sdma1 reset failed
amdgpu 0000:09:00.0: GPU reset begin!. Source:  1
amdgpu 0000:09:00.0: GPU reset end with ret = -19
amdgpu 0000:09:00.0: GPU Recovery Failed: -19
```

`-19` is `ENODEV`: the driver could not reach the device at all. The ring timeouts came after the link was already
gone, so they are a symptom, not the trigger.

## State after the drop

- `lspci -s 09:00.0`: `!!! Unknown header type 7f` (config space reads return `0xFF`); the card's internal PCIe
  switch (`07:00.0` upstream, `08:00.0` downstream) was unreadable too.
- `current_link_speed`: `Unknown`; `current_link_width`: `63`. Root port `00:1b.4` had retrained down from its
  16 GT/s capability to **2.5 GT/s x4** (`Target Link Speed` still 16 GT/s).
- `devcoredump/data`: **0 bytes** - the driver could not capture device state; the failing-device link pointed at
  `0000:09:00.0`.
- Reset loop: 82 `GPU Recovery Failed`, 6,700+ `device lost from bus` lines in ~10 min, still ongoing when the host
  was rebooted.
- Software removal (`.../remove`) blocked in uninterruptible sleep; `nvtop` also stuck in D state.

## What it was not

- **Not a maya/model bug**: the old container's health check passed 10 s before the first AER, and the failure
  signature is below the driver/engine layer.
- **Not a GPU engine hang**: no MES/ring timeout preceded the link error; the AER RxErr came first, and the empty
  devcoredump shows the driver never got to read GPU state.
- **Not thermal/OOM/power logging**: `journalctl -k` had no thermal throttle, OOM, MCE or power-event messages
  before 21:39:17. After recovery both cards read normal junction temperatures (52-74 C).
- **Not recurring (so far)**: no `device lost from bus`, `GPU Recovery Failed`, ring timeout or AER error in any
  boot from 2026-10-04 through 2026-11-09 before this event.

## Recovery

1. Host restarted; the riser and both 8-pin power connectors were checked (visually normal).
2. At POST the link retrained at **16 GT/s x4** on the PCH root port; endpoint at 16 GT/s x16 (its internal link).
3. Zero AER/amdgpu errors since boot; GLM lane served a 144,766-token request at 15.3 tok/s with no link events.

## Assessment

The only electrical evidence is the **root-port receiver error** (`RxErr`, physical layer) one line before the device
vanished. That points at the link path - riser, connectors, or the card's PHY - rather than at the GPU cores or the
serving stack. The kernel saw exactly what it would see if the card had been unplugged mid-operation.

Prime suspect is **Gen4 link margin on the 70 cm bundled riser**. A cable marketed as "PCIe 4.0 x16" does not
guarantee margin at 16 GT/s, and 70 cm is electrically long; only 4 lanes are used in this slot, so a single marginal
lane/contact is enough. A transient power draw from the 355 W card on a marginal slot/connector contact is the other
plausible trigger. Nothing was visibly wrong and a restart restored full speed, so this remains **suspected link
marginality, not a confirmed root cause** - first occurrence, watch for repeats.

## If it recurs

1. Record the time and capture `journalctl -k -b 0 | grep -B2 -A2 -E 'AER:|device lost from bus'`,
   `lspci -s 09:00.0 -vv`, and `cat /sys/class/drm/card0/device/current_link_speed`.
2. Full power-off + reseat both riser ends and the power connectors; do **not** rely on a warm reboot - the link
   will not retrain while the card stays wedged.
3. Swap in a known-good (preferably shorter) Gen4 riser. To separate card / riser / slot, test the card in the CPU
   x16 slot (`PCIEX16(G5)`) and, if possible, the riser with the other card.
4. As a diagnostic, set `PCIEX16(G4)_2` to Gen3 in BIOS and run load for a while. Stable at Gen3 but failing at Gen4
   confirms signal margin. Gen3 x4 (~3.9 GB/s) is not a long-term option: expert streaming and pp depend on the
   Gen4 x4 bandwidth.
5. If the card is not enumerated even in the CPU slot after a cold boot, treat it as a card failure.

## Monitoring

Cheap watchdog while the riser is suspect:

```sh
journalctl -k -f | grep -m1 -E 'AER:.*(error|Corrected)|device lost from bus' && notify-send 'GPU link error'
```

Per-boot health check:

```sh
for c in card0 card1; do echo -n "$c "; cat /sys/class/drm/$c/device/current_link_speed; done
sudo lspci -s 00:1b.4 -vv | grep LnkSta:        # expect Speed 16GT/s, Width x4
sudo journalctl -k -b 0 | grep -cE 'AER:.*error|device lost from bus|GPU Recovery Failed'
```

---

# 20261116 Second drop of GPU1 (`09:00.0`): identical signature, mid-prefill on the 1.1.65 A/B ladder

## Summary

Seven days after the 20261109 incident, the same PCH-attached x4 card (`0000:09:00.0`, DRM `card0`, maya's GPU1)
dropped off the bus again with the **identical kernel opening**: a correctable physical-layer RxErr at root port
`00:1b.4`, then `device lost from bus!` one line later (17:52:35 CST). The card died mid-prefill during the Maya
1.1.65 verification ladder, so this incident also documents how the drop looks through maya's watchdogs. The
20261109 "not recurring (so far)" line is superseded: it recurred, and the runbook's next diagnostics (riser swap,
Gen3 test) are warranted.

## Timeline (CST)

| Time | Event |
| --- | --- |
| 17:38:20 | GLM tune-ladder point `b65-r2` (Maya 1.1.65 arm) starts; its 40 % point completes normally |
| ~17:47 | its 85 % prompt starts prefilling 445,692 tokens |
| 17:49-17:52 | prefill reaches 311,296 tokens (~70 %); chunk walls degrade (19.7 s max vs 16.5-16.9 s on the other 7 points, mixer windows 14,394 vs ~11 k) |
| **17:52:35** | **AER correctable Physical-Layer RxErr (`[0] RxErr (First)`) at root port `00:1b.4`, then `amdgpu 0000:09:00.0: device lost from bus!`** |
| 17:55-17:56 | the engine's serve watchdog (#29) fires after 180 s of no progress at "the prompt: layer 23"; the engine stops itself (exit code 3) so the server restarts it; the in-flight request gets a 503 naming the exit code, the reason and the log line |
| 18:00:24 | the driver's own recovery cascade: SMU bus errors (`TransferTableSmu2Dram(18)` / `DisallowGfxOff(41)` response 0xFFFFFFFF), `ring sdma0 timeout`, sdma queue reset, devcoredump created |
| at writing (18:02) | reset loop ongoing: 7,247 AER / device-lost / recovery-failed lines this boot, ~830/minute |

## State after the drop

- `lspci -s 09:00.0` still enumerates the device and its internal switch bridges (`07:00.0`/`08:00.0`) — unlike
  20261109's `Unknown header type 7f` — but `current_link_speed: Unknown`, `current_link_width: 63`.
- Root port `00:1b.4`: `LnkSta: Speed 2.5GT/s, Width x4` (capability 16 GT/s; equalization incomplete) — the same
  degraded retrain as 20261109.
- `card1` (`03:00.0`, CPU x16, display): healthy, 16.0 GT/s, unaffected.
- `devcoredump/data`: **0 bytes** again (the driver never read GPU state).
- 24,267 MiB stuck as "used" on card0 with **no container or process holding it** (all maya/strata lanes were
  stopped): the killed engine's buffer objects are unreclaimable while the driver is wedged — D-state `ttm`
  kworkers (`u100`/`u101`).

## How the drop looks through maya (new this time)

- The engine hung inside a GPU call that never returned; nothing in the engine log precedes the link error. The
  serve watchdog (#29) then fired at +180 s ("no progress for 180 s during a request (the prompt: layer 23)"),
  the engine stopped itself (exit code 3), and the server reported the request with the exit code, the reason
  ("stopped itself because it had stopped making progress - a hang it caught") and the stage/position. On
  20261109 the equivalent event was an opaque lane hang at weight loading.
- The #860 drafting-stall watch did not fire (this was a prompt-path stall, not drafting) and the #899 pool-check
  wait was not involved (startup had long completed).
- **Not a Maya 1.1.65 regression**: the same ladder's completed points measured parity (b65-r1 85 % pp 978.5 /
  tg 18.4 vs the 1.1.64 arms' 967.1 / 16.75; all DS points within the run-to-run spread), and the GLM bits probes
  were byte-identical across all 4 points including across versions (`9eec69b5…`, the same hash as the 20261116
  1.1.64 round). The ladder's missing cell (b65-r2 85 %) is re-run after recovery.

## What it was not

- **Not thermal/OOM/power**: zero thermal-throttle, MCE, machine-check or OOM messages in the kernel log from
  17:00 until the 17:52:35 RxErr (checked; same method as 20261109).
- **Not a software/engine hang taking the link down**: prefill was progressing normally (chunks completing at
  ~1,000 t/s) until the RxErr; the engine-side stall began only when the card vanished.

## Recovery

Done the same evening, without hardware changes: force power-off, boot. The link retrained at POST — root port
`00:1b.4` back at **16 GT/s x4**, both cards 16 GT/s, **0** AER/amdgpu errors this boot, VRAM floors clean
(26/367 MiB desktop), llama-swap serving again. The 20261109 warning held: a warm reboot would not have done it.

Constraints recorded for the runbook's next diagnostics: no shorter riser is on hand (no spare cable; 50 cm does
not reach this layout at all, 70 cm barely fits), and the Gen3 diagnostic is rejected for now — expert streaming
and pp depend on the Gen4 x4 bandwidth. So steps 3-4 of the 20261109 runbook are not currently actionable;
monitoring (the per-boot check and the `journalctl -f` watchdog) continues.

Working hypotheses for the repeat (user's, unverified — recorded because they shape what to watch):

- **Ladder-class load**: both drops happened during long-prefill streaming (20261109: a 144k-token production
  prefill; 20261116: a 445k-token ladder prompt, the heaviest sustained PCIe-x4 + disk streaming this host runs).
  Short chats and idle never trigger it. If drops only ever follow ladder-class prefill, load-correlated margin
  (power draw transient on the riser/connector under sustained reads) stays the prime suspect.
- **Load/unload cycles tickling a driver bug**: the ladders also cycle engine boots/stops every ~20 minutes
  (llama-swap restarts, expert-tier pinning, VRAM lend/return); a TTM/GEM- or HIP-level bug could in principle
  wedge the card. Counter-evidence: the kernel opening is a physical-layer receiver error at the root port —
  below the driver — and there is no MCE/thermal/OOM precursor; but a driver-side wedge and an electrical margin
  problem are not yet fully separable from this host alone.

What would discriminate: a drop on a swapped (shorter) riser → card; a drop at Gen3 under the same load → not
signal margin alone; no drop ever outside ladder-class prefill → load-correlated; a drop while idle at desktop
load → card or slot power. Until then: keep ladders away from production hours where possible, watch the first
minutes of any 85 % ladder prompt (both drops sat inside or right at sustained streaming), and treat a third
occurrence as the trigger to take the slot out of service pending the riser swap.

## Assessment

Two identical drops seven days apart on the same 70 cm riser move the 20261109 "suspected link marginality" from
a one-off towards a confirmed margin problem on the PCH x4 slot path (riser cable, connectors, or the card's PHY
at Gen4). The degradation pattern before both drops (none visible last time; chunk-wall creep this time) and the
identical root-port retrain behaviour point at the link, not the GPU cores. Until the riser is swapped or the
Gen3 test discriminates, treat this slot as suspect under sustained Gen4 load (the ladder's 85 % prompts are the
heaviest sustained PCIe+disk streaming this host runs).
