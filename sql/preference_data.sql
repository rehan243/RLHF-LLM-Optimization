-- human prefs and model scores; keep audit trail for later legal review

CREATE TABLE IF NOT EXISTS preference_pairs (
  id             BIGSERIAL PRIMARY KEY,
  prompt_id      UUID NOT NULL,
  chosen_text    TEXT NOT NULL,
  rejected_text  TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  source         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pref_prompt ON preference_pairs (prompt_id);

CREATE TABLE IF NOT EXISTS annotator_labels (
  id          BIGSERIAL PRIMARY KEY,
  pair_id     BIGINT NOT NULL REFERENCES preference_pairs (id) ON DELETE CASCADE,
  annotator   TEXT NOT NULL,
  label       TEXT NOT NULL CHECK (label IN ('A', 'B', 'tie', 'bad')),
  comment     TEXT,
  labeled_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_labels_pair ON annotator_labels (pair_id);

CREATE TABLE IF NOT EXISTS annotator_agreement (
  pair_id    BIGINT PRIMARY KEY REFERENCES preference_pairs (id) ON DELETE CASCADE,
  agree_cnt  INTEGER NOT NULL,
  total_cnt  INTEGER NOT NULL,
  kappa      DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS reward_model_scores (
  id         BIGSERIAL PRIMARY KEY,
  pair_id    BIGINT NOT NULL REFERENCES preference_pairs (id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  chosen_score    DOUBLE PRECISION NOT NULL,
  rejected_score  DOUBLE PRECISION NOT NULL,
  margin          DOUBLE PRECISION GENERATED ALWAYS AS (chosen_score - rejected_score) STORED,
  scored_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rm_pair ON reward_model_scores (pair_id);
CREATE INDEX IF NOT EXISTS idx_rm_model ON reward_model_scores (model_name, scored_at DESC);

-- helps find pairs where humans agree but reward model flips sign
CREATE INDEX IF NOT EXISTS idx_rm_margin
  ON reward_model_scores (margin);

-- fast lookup when you only care about recent human labels for nightly jobs
CREATE INDEX IF NOT EXISTS idx_labels_recent
  ON annotator_labels (labeled_at DESC)
  WHERE label <> 'bad';
