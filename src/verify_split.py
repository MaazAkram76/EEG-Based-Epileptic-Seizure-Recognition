import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

FILENAME = 'data.csv'

def verify_split():
    print(f"📂 Loading {FILENAME}...")
    try:
        df = pd.read_csv(FILENAME)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    ids = df.iloc[:, 0]
    y = df.iloc[:, -1].values
    
    # ---------------------------------------------------------
    # ID PARSING LOGIC
    # ---------------------------------------------------------
    def parse_id(uid):
        try:
            parts = uid.split('.')
            last = parts[-1]
            if last.startswith('V'):
                last = last[1:]
            return int(last)
        except:
            return -1

    def parse_prefix(uid):
        try:
            # X21 -> 21
            return int(uid.split('.')[0].replace('X',''))
        except:
            return -1

    print("🔍 Parsing IDs...")
    raw_indices = ids.astype(str).apply(parse_id)
    prefixes = ids.astype(str).apply(parse_prefix)
    
    # ---------------------------------------------------------
    # HYPOTHESIS TESTING
    # ---------------------------------------------------------
    print(f"   -> Min Suffix: {raw_indices.min()}")
    print(f"   -> Max Suffix: {raw_indices.max()}")
    
    # Check simple suffix grouping
    # groups_desc = (raw_indices - 1) // 23
    # check_consistency(groups_desc, y, "Simple Suffix")

    # Check Class+Suffix Grouping
    print("\n🧪 Testing Group = (Class, Suffix)")
    # Since Class (y) is 1..5, we can form unique ID easily
    # GroupID = y * 10000 + Suffix
    groups = y * 10000 + raw_indices
    
    unique_groups = np.unique(groups)
    print(f"   -> Total Unique Groups: {len(unique_groups)}")
    
    # Calculate group sizes
    df_g = pd.DataFrame({'group': groups})
    sizes = df_g.groupby('group').size()
    print(f"   -> Avg Size: {sizes.mean():.2f}")
    print(f"   -> Median Size: {sizes.median()}")
    print(f"   -> Min Size: {sizes.min()}")
    print(f"   -> Max Size: {sizes.max()}")
    
    # Consistency Check (Trivial for Class+Suffix, but good to run)
    check_df = pd.DataFrame({'group': groups, 'label': y})
    inc = check_df.groupby('group')['label'].nunique()
    if inc.max() > 1:
        print("❌ LEAKAGE detected in hypothesis!")
    else:
        print("✅ No Leakage (by definition logic).")

    # ---------------------------------------------------------
    # PERFORM SPLIT
    # ---------------------------------------------------------
    print("\n✂️ Performing GroupShuffleSplit...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(ids, y, groups=groups))
    
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    
    print(f"   -> Train Samples: {len(train_idx)}")
    print(f"   -> Test Samples: {len(test_idx)}")
    
    intersect = train_groups.intersection(test_groups)
    if len(intersect) == 0:
        print("✅ SPLIT VALIDATION PASSED: Subject disjoint.")
    else:
        print(f"❌ SPLIT FAILED: Overlap found.")

if __name__ == "__main__":
    verify_split()
