CREATE TABLE IF NOT EXISTS nfl_prospects (
    id SERIAL PRIMARY KEY,

    -- Identity
    pfr_id          VARCHAR(20)  UNIQUE,
    cfb_id          VARCHAR(20),
    gsis_id         VARCHAR(20),
    player_name     VARCHAR(100) NOT NULL,
    position        VARCHAR(10),
    draft_year      SMALLINT,
    draft_round     SMALLINT,
    draft_pick      SMALLINT,
    draft_team      VARCHAR(5),
    college         VARCHAR(100),

    -- Combine measurables (all nullable)
    height_in       NUMERIC(4,1),
    weight_lbs      SMALLINT,
    forty_yard      NUMERIC(4,2),
    vertical_jump   NUMERIC(4,1),
    broad_jump      SMALLINT,
    bench_reps      SMALLINT,
    cone_drill      NUMERIC(4,2),
    shuttle         NUMERIC(4,2),

    -- College career totals: passing
    col_pass_completions  INTEGER,
    col_pass_attempts     INTEGER,
    col_pass_yards        INTEGER,
    col_pass_tds          INTEGER,
    col_pass_ints         INTEGER,

    -- College career totals: rushing
    col_rush_attempts     INTEGER,
    col_rush_yards        INTEGER,
    col_rush_tds          INTEGER,

    -- College career totals: receiving
    col_receptions        INTEGER,
    col_rec_yards         INTEGER,
    col_rec_tds           INTEGER,

    -- College career totals: defense
    col_total_tackles     INTEGER,
    col_tfl               NUMERIC(5,1),
    col_sacks             NUMERIC(5,1),
    col_ints              INTEGER,
    col_pass_deflections  INTEGER,
    col_qb_hurries        INTEGER,

    -- College career totals: kicking
    col_fg_made           INTEGER,
    col_fg_attempted      INTEGER,
    col_xp_made           INTEGER,
    col_xp_attempted      INTEGER,

    -- College career totals: punting
    col_punts             INTEGER,
    col_punt_yards        INTEGER,

    -- Target variables: Career AV (from Pro Football Reference via nflverse)
    career_av       INTEGER,
    weighted_av     NUMERIC(6,1),
    draft_team_av   INTEGER,

    -- Additional NFL career context
    nfl_games       INTEGER,
    pro_bowls       SMALLINT,
    all_pro         SMALLINT,
    seasons_started SMALLINT,
    hof             BOOLEAN,

    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nfl_prospects_draft_year ON nfl_prospects (draft_year);
CREATE INDEX IF NOT EXISTS idx_nfl_prospects_position   ON nfl_prospects (position);
CREATE INDEX IF NOT EXISTS idx_nfl_prospects_cfb_id     ON nfl_prospects (cfb_id);
