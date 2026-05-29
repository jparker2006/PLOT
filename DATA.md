# Data — sources, provenance, and availability

PLOT is built on **public 2015-16 NBA SportVU player-tracking data** (25 fps x/y of 10 players +
3D ball) merged with play-by-play. 2015-16 is the **last season** for which league tracking data
was made publicly available; the data was distributed through community archives after the
`stats.nba.com` movement endpoint was closed.

## Stance on redistribution

This repository **does not re-host the raw tracking data.** The NBA never issued an official
public release of it. Instead we ship:

- **code** to download the data from existing public archives,
- **derived/aggregated features and model outputs** needed to reproduce results and figures,
- documentation of every transformation.

If you are a rights-holder and want a source removed from the download script, please open an
issue.

## Sources

| Source | What | Use here |
|---|---|---|
| [`dcayton/nba_tracking_data_15_16`](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16) | Restructured SportVU @ 25 fps, with play-by-play fields merged into `event_info`; configs `tiny`(5)/`small`(25)/`medium`(100)/`large`(600+) | **Primary** source for bulk games + PBP alignment |
| [`linouk23/NBA-Player-Movements`](https://github.com/linouk23/NBA-Player-Movements) | Raw per-game SportVU JSON (`Away@Home.json`) + matplotlib animation | Fetch **named** demo games; animation reference |
| [`neilmj/BasketballData`](https://github.com/neilmj/BasketballData) | Raw SportVU logs | Backup raw source |
| [`nba_api`](https://github.com/swar/nba_api) | Box scores, shot charts, dashboards | Supplementary priors only, if needed |

## Known data-quality issues (handle in the pipeline)

- Some events / moments have **no coordinates** (small fraction of events).
- Tracking coordinates for an event often **start before and/or end after** the labeled event —
  must be trimmed on possession boundaries.
- Tracking ↔ play-by-play misalignment occurs on some plays; verify the `game_clock` join early.

## Schema (as advertised by `dcayton`, to be verified in Stage 1)

```
game: { gameid, gamedate, events: [ {
  event_info: { eventid, type, possession_team_id, desc_home, desc_away },
  home/visitor: { name, teamid, abbreviation, players: [...] },
  moments: [ {
    quarter, game_clock (720→0), shot_clock (24→0),
    ball_coordinates: {x, y, z},
    player_coordinates: [ {teamid, playerid, x, y, z} x10 ]
  } ]
} ] }
```

Coordinates are in **feet**; full court is 94 × 50.
