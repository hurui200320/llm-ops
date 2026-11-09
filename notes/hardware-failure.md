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
