# tests/gx_audio_metadata_validation.py
from pathlib import Path
import pandas as pd
import great_expectations as gx
from great_expectations.expectations.core import (
    ExpectColumnPairValuesToBeInSet,
    ExpectTableRowCountToBeBetween,
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToMatchRegex,
    ExpectColumnValuesToBeUnique,
    ExpectCompoundColumnsToBeUnique,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "processed" / "audio_metadata.csv"

# Context
context = gx.get_context(mode="file", project_root_dir=ROOT)
ds = context.data_sources.add_or_update_pandas(name="audio_metadata")
asset = ds.add_csv_asset(name="metadata_csv", filepath_or_buffer=str(CSV_PATH))
batch_def = asset.add_batch_definition(name="full")

# Load data
df = pd.read_csv(CSV_PATH)

def has_cols(*cols) -> bool:
    return all(c in df.columns for c in cols)

suite = gx.ExpectationSuite("audio_metadata_validation")
context.suites.add_or_update(suite)

# General checks
suite.add_expectation(ExpectTableRowCountToBeBetween(min_value=1200, max_value=2000))

# Key columns not null
for col in [
    "filename","relative_path","modality","vocal_channel",
    "emotion","emotion_code","intensity","intensity_code",
    "statement","statement_code","repetition",
    "actor","actor_code","valid"
]:
    if has_cols(col):
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

# Unicity checks
if has_cols("filename"):
    suite.add_expectation(ExpectColumnValuesToBeUnique(column="filename"))
if has_cols("relative_path"):
    suite.add_expectation(ExpectColumnValuesToBeUnique(column="relative_path"))

combo = ["actor_code","statement","emotion","intensity","repetition"]
if has_cols(*combo):
    suite.add_expectation(ExpectCompoundColumnsToBeUnique(column_list=combo))

# emotions
emotions = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
if has_cols("emotion"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="emotion", value_set=emotions))

# emotion_code 
if has_cols("emotion_code"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(
        column="emotion_code",
        value_set=["01","02","03","04","05","06","07","08",1,2,3,4,5,6,7,8]
    ))

# intensity & intensity_code
if has_cols("intensity"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="intensity", value_set=["normal","strong"]))
if has_cols("intensity_code"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="intensity_code", value_set=["01","02",1,2]))

# statement & statement_code
statements = ["Kids are talking by the door","Dogs are sitting by the door"]
if has_cols("statement"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="statement", value_set=statements))
if has_cols("statement_code"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="statement_code", value_set=["01","02",1,2]))

# repetition (1 o 2)
if has_cols("repetition"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="repetition", value_set=["01","02",1,2]))

# modality / vocal_channel
if has_cols("modality"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="modality", value_set=["03",3]))
if has_cols("vocal_channel"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="vocal_channel", value_set=["01",1]))

# actor & actor_code
if has_cols("actor_code"):
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="actor_code", min_value=1, max_value=24))
if has_cols("actor"):
    suite.add_expectation(ExpectColumnValuesToMatchRegex(column="actor", regex=r"^Actor_\d{2}$"))

# valid booleano
if has_cols("valid"):
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="valid", value_set=[True, False]))

# Consistency between columns
# emotion <-> emotion_code
allowed_emotion_pairs = [
    ("neutral","01"),("neutral",1),
    ("calm","02"),("calm",2),
    ("happy","03"),("happy",3),
    ("sad","04"),("sad",4),
    ("angry","05"),("angry",5),
    ("fearful","06"),("fearful",6),
    ("disgust","07"),("disgust",7),
    ("surprised","08"),("surprised",8),
]
if has_cols("emotion","emotion_code"):
    suite.add_expectation(ExpectColumnPairValuesToBeInSet(
        column_A="emotion", column_B="emotion_code", value_pairs_set=allowed_emotion_pairs
    ))

# intensity <-> intensity_code
allowed_intensity_pairs = [("normal","01"),("normal",1),("strong","02"),("strong",2)]
if has_cols("intensity","intensity_code"):
    suite.add_expectation(ExpectColumnPairValuesToBeInSet(
        column_A="intensity", column_B="intensity_code", value_pairs_set=allowed_intensity_pairs
    ))

# statement <-> statement_code
allowed_statement_pairs = [("Kids are talking by the door","01"),("Kids are talking by the door",1),
                           ("Dogs are sitting by the door","02"),("Dogs are sitting by the door",2)]
if has_cols("statement","statement_code"):
    suite.add_expectation(ExpectColumnPairValuesToBeInSet(
        column_A="statement", column_B="statement_code", value_pairs_set=allowed_statement_pairs
    ))

# actor <-> actor_code
if has_cols("actor","actor_code"):
    pairs = [(f"Actor_{i:02d}", i) for i in range(1,25)] + [(f"Actor_{i:02d}", f"{i:02d}") for i in range(1,25)]
    suite.add_expectation(ExpectColumnPairValuesToBeInSet(
        column_A="actor", column_B="actor_code", value_pairs_set=pairs
    ))

# Archive patterns
if has_cols("filename"):
    suite.add_expectation(ExpectColumnValuesToMatchRegex(column="filename", regex=r".+\.wav$"))
if has_cols("relative_path"):
    # Should start with Actor_XX/ and end in .wav formats
    suite.add_expectation(ExpectColumnValuesToMatchRegex(column="relative_path", regex=r"^Actor_\d{2}/.+\.wav$"))

suite.save()

# Validation + Checkpoint
vd = gx.ValidationDefinition(name="audio_metadata_validator", data=batch_def, suite=suite)
context.validation_definitions.add_or_update(vd)

checkpoint = gx.Checkpoint(
    name="audio_metadata_checkpoint",
    validation_definitions=[vd],
    actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
    result_format="SUMMARY",
)
context.checkpoints.add_or_update(checkpoint)

if __name__ == "__main__":
    print("▶ Running checkpoint…")
    res = context.checkpoints.get("audio_metadata_checkpoint").run()
    print(f"Done. Expectations in suite: {len(suite.expectations)}")
    print("Open: gx/uncommitted/data_docs/local_site/index.html")
