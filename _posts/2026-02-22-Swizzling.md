---
layout: post
title: "Decoding Swizzle&lt;B,M,S&gt;: A Visual Guide to Bank-Conflict-Free Shared Memory Access"
author: "Vijay Krishnamoorthy"
tags: [GPU, SMEM, Swizzling]
---

## Overview

Efficient use of shared memory (SMEM) is critical for achieving peak performance in GPU kernels. One of the most common pitfalls, bank conflicts, can silently degrade performance by forcing serialized memory accesses. Swizzling is a powerful technique that reorganizes data layout to eliminate these conflicts. This article builds a mental model for understanding swizzling, culminating in a thorough explanation of CuTe's canonical `Swizzle<BBits, MBase, SShift>` representation.

## 1. The Problem: Bank Conflicts

### How Shared Memory Banks Work

GPU shared memory is organized into **banks** — interleaved partitions that can each service one memory access per cycle. Think of banks as parallel lanes: when different threads access different banks, all accesses proceed simultaneously. On modern NVIDIA GPUs:

- SMEM is divided into **32 banks**
- Each bank is **4 bytes (1 DWORD)** wide
- One SMEM "row" spans all 32 banks = **128 bytes**

```
SMEM Row (128 bytes):
Bank:  0    1    2    3    4   ...  30   31
     [4B] [4B] [4B] [4B] [4B] ... [4B] [4B]
```

For example, when 32 threads in a warp access 32 different banks simultaneously, all accesses complete in a single transaction. This is an ideal scenario.

### When Conflicts Occur

A **bank conflict** occurs when multiple threads access different addresses within the same bank. The hardware must serialize these accesses, degrading performance.

Consider a matrix stored in row-major order where each element is 4 bytes:

```
Logical Layout (8x8 matrix, 4 bytes per element):

          Col 0   Col 1   Col 2   Col 3   Col 4   Col 5   Col 6   Col 7
        +-------+-------+-------+-------+-------+-------+-------+-------+
Row 0   | B0    | B1    | B2    | B3    | B4    | B5    | B6    | B7    |
Row 1   | B0    | B1    | B2    | B3    | B4    | B5    | B6    | B7    |
Row 2   | B0    | B1    | B2    | B3    | B4    | B5    | B6    | B7    |
...     | ...   | ...   | ...   | ...   | ...   | ...   | ...   | ...   |
Row 7   | B0    | B1    | B2    | B3    | B4    | B5    | B6    | B7    |
        +-------+-------+-------+-------+-------+-------+-------+-------+
          ^       ^       ^       ^       ^       ^       ^       ^
          |       |       |       |       |       |       |       |
        Bank 0  Bank 1  Bank 2  Bank 3  Bank 4  Bank 5  Bank 6  Bank 7
```

**Row-major access (reading Row 0):** Each thread reads a different column → different banks → no conflict.

**Column-major access (reading Col 0):** Every thread reads from Bank 0 → **8-way bank conflict!**

## 2. The Solution: Swizzling

Swizzling reorganizes how data is stored in physical memory so that column-wise access patterns also hit different banks. The key insight is:

> **Apply a row-dependent transformation to the column index when storing data, so that elements in the same logical column end up in different physical banks.**

### The XOR Trick

The most elegant swizzling scheme uses bitwise XOR. For each element at logical position `(row, col)`:

```
physical_col = logical_col XOR row
```

Let's see this in action for an 8x8 tile:

```
Logical Layout:              Physical Layout (after XOR swizzle):

Col: 0 1 2 3 4 5 6 7         Col: 0 1 2 3 4 5 6 7
   +----------------+           +----------------+
R0 | 0 1 2 3 4 5 6 7 |       R0 | 0 1 2 3 4 5 6 7 |  (XOR 0 = no change)
R1 | 0 1 2 3 4 5 6 7 |       R1 | 1 0 3 2 5 4 7 6 |  (XOR 1)
R2 | 0 1 2 3 4 5 6 7 |       R2 | 2 3 0 1 6 7 4 5 |  (XOR 2)
R3 | 0 1 2 3 4 5 6 7 |       R3 | 3 2 1 0 7 6 5 4 |  (XOR 3)
R4 | 0 1 2 3 4 5 6 7 |       R4 | 4 5 6 7 0 1 2 3 |  (XOR 4)
R5 | 0 1 2 3 4 5 6 7 |       R5 | 5 4 7 6 1 0 3 2 |  (XOR 5)
R6 | 0 1 2 3 4 5 6 7 |       R6 | 6 7 4 5 2 3 0 1 |  (XOR 6)
R7 | 0 1 2 3 4 5 6 7 |       R7 | 7 6 5 4 3 2 1 0 |  (XOR 7)
   +----------------+           +----------------+

(Numbers show which logical column's data is stored at each physical position)
```

Now look at any physical column in the swizzled layout—say, physical column 0:

- Row 0: logical col 0
- Row 1: logical col 1
- Row 2: logical col 2
- Row 3: logical col 3
- Row 4: logical col 4
- Row 5: logical col 5
- Row 6: logical col 6
- Row 7: logical col 7

**Every row contains a different logical column!** When we read column 0 (rows 0-7), each thread accesses a different physical column → different banks → **no conflict**.

### Why XOR Works Mathematically

The XOR operation with row indices guarantees conflict-free access because:

1. **XOR is its own inverse:** `a XOR b XOR b = a`
2. **Unique mapping per row:** For any fixed logical column `c`, as row `r` varies from 0 to 2^n-1, the value `c XOR r` produces all values from 0 to 2^n-1 exactly once.
3. **Bijection:** XOR with a constant is a bijection (one-to-one mapping), so no two logical columns in the same row map to the same physical column.

## 3. CuTe's Canonical Swizzle Form

NVIDIA's CuTe library provides a canonical representation for swizzle patterns:

```cpp
Swizzle<BBits, MBase, SShift>
```

This compact notation encodes everything needed to describe a swizzle pattern. Before diving into each parameter, let's understand the high-level formula:

```
Given a byte address A, decompose it into bit fields:
```

```
A = [ ... ] [ YYY ] [ ... ] [ ZZZ ] [ ... ]
       ↑       ↑       ↑       ↑       ↑
       │       │       │       │       │
       │       │       │       │       └── bits [0, MBase)
       │       │       │       │
       │       │       │       └── bits [MBase, MBase+BBits)
       │       │       │
       │       │       └── bits [MBase+BBits, MBase+SShift)
       │       │
       │       └── bits [MBase+SShift, MBase+SShift+BBits)
       │
       └── bits [MBase+SShift+BBits, ...)

```

```
Swizzled address A' = A with ZZZ replaced by (ZZZ XOR YYY)
```

Now let's build intuition for what each parameter means.

## 4. Building Intuition: MBase, BBits, and SShift

While CuTe uses the ordering `<BBits, MBase, SShift>`, it's pedagogically clearer to understand them in a different order: **MBase → BBits → SShift**. Let's explore each.

### 4.1 MBase: The Swizzle Unit

**MBase** defines the fundamental unit of data that moves together during swizzling. The swizzle unit size is:

```
Swizzle Unit Size = 2^MBase bytes
```

**Key insight:** Data _within_ a swizzle unit is never rearranged. Swizzling only determines which physical slot a swizzle unit occupies—it doesn't shuffle bytes inside the unit.

| MBase | Swizzle Unit Size | Banks Spanned |
| ----- | ----------------- | ------------- |
| 2     | 4 bytes           | 1 bank        |
| 3     | 8 bytes           | 2 banks       |
| 4     | 16 bytes          | 4 banks       |
| 5     | 32 bytes          | 8 banks       |

**Why does this matter?** If your access pattern naturally reads 16-byte chunks (e.g., loading 4 fp32 values or 8 fp16 values per thread), you want MBase=4. There's no benefit to swizzling at a finer granularity—it would just add complexity.

### 4.2 BBits: The Swizzle Tile Dimensions

**BBits** defines the swizzle tile as a square grid:

```
Swizzle Tile = 2^BBits rows × 2^BBits columns (of swizzle units)
```

| BBits | Tile Shape | Swizzle Units per Tile |
| ----- | ---------- | ---------------------- |
| 1     | 2 × 2      | 4                      |
| 2     | 4 × 4      | 16                     |
| 3     | 8 × 8      | 64                     |

### 4.3 SShift: The Row Stride

Before diving into address bits, let's understand what SShift represents. In a row-major layout:

- Each row contains **2^SShift swizzle units**
- The row stride (bytes per row) is **2^(MBase+SShift)** bytes

Physically, SShift determines how wide each row is in memory. For a standard 128-byte SMEM row with 16-byte swizzle units:
- Row width = 128 bytes = 8 units × 16 bytes/unit
- SShift = log₂(8) = 3

Intuitively, **SShift encodes where the row index bits start** in the address. The column index occupies bits [MBase, MBase+SShift), and the row index starts at bit MBase+SShift.

A row-major address is structured as below -

```
Address = row × row_stride + col × unit_size
        = row × 2^(MBase+SShift) + col × 2^MBase
```

**Address bits map to a 3-level hierarchy:**

```
┌─ Level 1: SMEM Grid ──────────────────────────────────────────────────────┐
│  (swizzle tiles arranged in rows and columns)                             │
│                                                                           │
│  row_tile_idx                            col_tile_idx                     │
│       │                                       │                           │
│       │      ┌────────────┬────────────┬──────▼─────┬────────────┐        │
│       └─────►│  Tile 0,0  │  Tile 0,1  │  Tile 0,2  │  Tile 0,3  │ ...    │
│              ├────────────┼────────────┼────────────┼────────────┤        │
│              │  Tile 1,0  │  Tile 1,1  │  Tile 1,2  │  Tile 1,3  │ ...    │
│              └────────────┴────────────┴──────┬─────┴────────────┘        │
│                                               │                           │
│                                               ▼                           │
├─ Level 2: Inside a Tile ──────────────────────────────────────────────────┤
│  (2^BBits rows × 2^BBits columns of swizzle units)                        │
│                                                                           │
│  row_in_tile (YYY)                       col_in_tile (ZZZ)                │
│       │                                       │                           │
│       │        C0     C1     C2     C3     C4 ▼  C5     C6     C7         │
│       │      ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │
│       └─► R0 │     │     │     │     │  ●  │     │     │     │            │
│           R1 │     │     │     │     │     │     │     │     │            │
│           R2 │     │     │     │     │     │     │     │     │            │
│          ... └─────┴─────┴─────┴─────┴──┬──┴─────┴─────┴─────┘            │
│                                         │                                 │
│  ★ SWIZZLE: physical_col = col_in_tile XOR row_in_tile                    │
│                                         ▼                                 │
├─ Level 3: Inside a Swizzle Unit ──────────────────────────────────────────┤
│  (2^MBase bytes — NEVER modified by swizzling)                            │
│                                                                           │
│  byte_offset ──►  ┌────┬────┬────┬────┬────┬────┬─────┬──────────────┐    │
│                   │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ ... │ B(2^MBase-1) │    │
│                   └────┴────┴────┴────┴────┴────┴─────┴──────────────┘    │
│                                                                           │
│  Data within a swizzle unit always stays together.                        │
└───────────────────────────────────────────────────────────────────────────┘
```

Note: `col_tile_idx` only exists when SShift > BBits (multiple swizzle tiles per grid row of tiles). When SShift = BBits, the entire column index fits in `col_in_tile` and there's only one tile per row.

The swizzle operation XORs the YYY bits (row portion) into the ZZZ bits (column portion):

```
Swizzled Address = Original Address with ZZZ replaced by (ZZZ XOR YYY)
```

**Why this works:** The YYY bits contain the lower BBits of the row index. By XORing them into the ZZZ bits (lower BBits of column), we effectively remap which physical column stores each logical column—and the remapping is different for each row.

**The mask:** `mask = (1 << BBits) - 1` isolates BBits bits. For BBits=3: `mask = 0b111 = 7`

**Equivalence to row/column XOR:** Because of how the address is structured, the swizzle is _equivalent_ to:

```
col_in_tile = col & mask           // Lower BBits of column (ZZZ)
row_in_tile = row & mask           // Lower BBits of row (YYY)
physical_slot_in_tile = col_in_tile XOR row_in_tile
```

But remember: the actual operation is on address bits, not indices. The row bits at position [MBase+SShift, MBase+SShift+BBits) are XORed into the column bits at position [MBase, MBase+BBits).

**Why square tiles?** The XOR trick requires 2^BBits unique values in both YYY and ZZZ to create a bijection. With BBits rows and BBits column positions within a tile, we get exactly the right range of XOR operands.

To recap,

**SShift** specifies how many swizzle units fit in one logical row:

```
Swizzle Units per Row = 2^SShift
```

This parameter connects the swizzle tile to the actual SMEM layout. Typically:

```
SShift = log2(SMEM_row_bytes / swizzle_unit_bytes)
       = log2(128 / 2^MBase)
       = 7 - MBase
```

| MBase | Swizzle Unit | SShift (for 128B SMEM row) | Units per Row |
| ----- | ------------ | -------------------------- | ------------- |
| 4     | 16 bytes     | 3                          | 8             |
| 5     | 32 bytes     | 2                          | 4             |

**Critical constraint:** `SShift >= BBits`

The swizzle tile width (2^BBits units) must fit within one row (2^SShift units). If the tile is wider than the row, the swizzle pattern breaks.

**Repetition:** When `SShift > BBits`, multiple swizzle tiles fit horizontally in one SMEM row:

```
Tiles per Row = 2^(SShift - BBits)
```

**Important:** Swizzling operates independently within each tile. There's no swizzling _across_ tile boundaries—each tile applies its own XOR pattern based on the row index _within that tile_.

### 4.4 Putting It All Together

For `Swizzle<BBits=3, MBase=4, SShift=3>`:

- **Swizzle unit:** 16 bytes (MBase=4), so MBase=4 means bits [0,4) are byte offset within unit
- **Swizzle tile:** 8×8 units = 8 rows × 128 bytes (BBits=3)
- **Units per row:** 8 (SShift=3)
- **Tiles per row:** 1 (SShift - BBits = 0)

**Address bit layout:**

```
Original Address (for row r, column c, byte offset b):
A = r × 2^(4+3) + c × 2^4 + b = r × 128 + c × 16 + b

Bit positions:
  [13..10]  [9  8  7]  [6  5  4]  [3  2  1  0]
  row_tile  row_in     col_in      byte
   _idx      _tile      _tile      _offset
             (YYY)      (ZZZ)

             ↑ bits [7,10)  ↑ bits [4,7)   ↑ bits [0,4)
```

Since SShift = BBits = 3, there is no `col_tile_idx` field—the entire column fits in `col_in_tile`. The swizzle tile spans the full SMEM row width.

**The swizzle operation:**

```
ZZZ = (Address >> MBase) & 0b111          // Extract bits [4,7) = col_in_tile
YYY = (Address >> (MBase+SShift)) & 0b111 // Extract bits [7,10) = row_in_tile

Swizzled Address = Address XOR (YYY << MBase)
                 = Address with ZZZ replaced by (ZZZ XOR YYY)
```

**Equivalent index-based view:**

```
col_in_tile = c & 0b111                   // Lower 3 bits of column
row_in_tile = r & 0b111                   // Lower 3 bits of row
physical_slot_in_tile = col_in_tile XOR row_in_tile
physical_col = (c & ~0b111) | physical_slot_in_tile  // Preserve col tile index
```

The key insight: we XOR the row bits (YYY) into the column bits (ZZZ) at their respective positions in the address. The upper column bits (tile index) and all other bits pass through unchanged, ensuring swizzling is self-contained within each tile.

## 5. Worked Examples

Let's walk through concrete examples using standard GPU SMEM configuration:

- **Bank size:** 4 bytes (1 DWORD)
- **Number of banks:** 32
- **SMEM row:** 128 bytes

We'll examine different swizzle configurations corresponding to different "swizzle atoms" from the previous post on MMA layouts.

### 5.1 Example A: No Swizzle — 8×16B (1 Core Matrix)

**Configuration:** `Swizzle<0, 4, 3>`

- BBits=0: No swizzle (tile is 1×1)
- MBase=4: 16-byte units
- SShift=3: 8 units per row

**Logical Layout (8 rows × 8 units of 16B each = 8 × 128B):**

```
        +--------+--------+--------+--------+--------+--------+--------+--------+
        | Tile 0 | Tile 1 | Tile 2 | Tile 3 | Tile 4 | Tile 5 | Tile 6 | Tile 7 |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 1   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 2   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 3   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 4   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 5   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 6   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
Row 7   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    |
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

**Physical Layout:** Same as logical (no swizzle applied)

**Bank Mapping (each 16B unit spans 4 banks):**

```
          Banks 0-3  Banks 4-7  Banks 8-11 Banks 12-15 Banks 16-19 Banks 20-23 Banks 24-27 Banks 28-31
        +----------+----------+----------+-----------+-----------+-----------+-----------+-----------+
Row 0   |  Unit 0  |  Unit 1  |  Unit 2  |  Unit 3   |  Unit 4   |  Unit 5   |  Unit 6   |  Unit 7   |
Row 1   |  Unit 0  |  Unit 1  |  Unit 2  |  Unit 3   |  Unit 4   |  Unit 5   |  Unit 6   |  Unit 7   |
...     |   ...    |   ...    |   ...    |   ...     |   ...     |   ...     |   ...     |   ...     |
        +----------+----------+----------+-----------+-----------+-----------+-----------+-----------+
```

**Column Access Pattern (reading logical Unit 0):**

Bank conflict across threads accessing the units in a column.

**Tensor Core access Pattern for swizzle atom with no swizzle (8x16B):**

No bank conflict.

### 5.2 Example B: 32B Swizzle Atom — 8×32B (2 Core Matrices)

**Configuration:** `Swizzle<1, 4, 3>`

- BBits=1: 2×2 tile of swizzle units
- MBase=4: 16-byte units
- SShift=3: 8 units per row

**Swizzle tile:** 2 rows × 2 units (32 bytes wide)

**XOR values by row:**

- Row 0, 2, 4, 6: XOR with 0 (row & 1 = 0)
- Row 1, 3, 5, 7: XOR with 1 (row & 1 = 1)

**Physical Layout (showing logical column at each physical position):**

```
        +-----------------+-----------------+-----------------+-----------------+
        |      Tile 0     |      Tile 1     |      Tile 2     |     Tile 3      |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 1   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 2   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 3   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 4   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 5   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 6   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 7   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
        +--------+--------+--------+--------+--------+--------+--------+--------+

```

**Column Access Pattern (reading logical Unit 0):**

- Row 0: Physical Unit 0 (Banks 0-3)
- Row 1: Physical Unit 1 (Banks 4-7)
- Row 2: Physical Unit 0 (Banks 0-3)
- Row 3: Physical Unit 1 (Banks 4-7)
- ...

**Tensor Core access Pattern for 2 core matrix wide swizzle atom (8x32B):**

No bank conflict. 

Without swizzling, the 2 core matrices would be stored across 2 SMEM rows resulting in bank conflicts when loading the 8x32B atom.

If you recall from the last post, a core matrix is 8x16B, so each row maps to a swizzle unit spanning 4 banks in this case.

```
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   R0   |   R1   |   R2   |   R3   |   R4   |   R5   |   R6   |   R7   | Core Matrix 0
Row 1   |   R0   |   R1   |   R2   |   R3   |   R4   |   R5   |   R6   |   R7   | Core Matrix 1
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

No bank-conflict layout with swizzling, 

```
        +-----------------+-----------------+-----------------+-----------------+
        |      Tile 0     |      Tile 1     |      Tile 2     |     Tile 3      |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   R0   |   R1   |   R2   |   R3   |   R4   |   R5   |   R6   |   R7   | Core Matrix 0
Row 1   |   R1   |   R0   |   R3   |   R2   |   R5   |   R4   |   R7   |   R6   | Core Matrix 1
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

### 5.3 Example C: 64B Swizzle Atom — 8×64B (4 Core Matrices)

**Configuration:** `Swizzle<2, 4, 3>`

- BBits=2: 4×4 tile of swizzle units
- MBase=4: 16-byte units
- SShift=3: 8 units per row

**Swizzle tile:** 4 rows × 4 units (64 bytes wide)

**XOR values by row:**

- Row 0, 4: XOR with 0
- Row 1, 5: XOR with 1
- Row 2, 6: XOR with 2
- Row 3, 7: XOR with 3

**Physical Layout:**

```
        +-----------------------------------+-----------------------------------+
        |               Tile 0              |               Tile 1              |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 1   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
Row 2   |   2    |   3    |   0    |   1    |   6    |   7    |   4    |   5    | XOR 2
Row 3   |   3    |   2    |   1    |   0    |   7    |   6    |   5    |   4    | XOR 3
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 4   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 5   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
Row 6   |   2    |   3    |   0    |   1    |   6    |   7    |   4    |   5    | XOR 2
Row 7   |   3    |   2    |   1    |   0    |   7    |   6    |   5    |   4    | XOR 3
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

**Column Access Pattern (reading logical Unit 0):**

- Row 0: Physical Unit 0 (Banks 0-3)
- Row 1: Physical Unit 1 (Banks 4-7)
- Row 2: Physical Unit 2 (Banks 8-11)
- Row 3: Physical Unit 3 (Banks 12-15)
- Row 4: Physical Unit 0 (Banks 0-3)
- Row 5: Physical Unit 1 (Banks 4-7)
- Row 6: Physical Unit 2 (Banks 8-11)
- Row 7: Physical Unit 3 (Banks 12-15)

**Tensor Core access Pattern for 4 core matrix wide swizzle atom (8x64B):**

No bank conflict. 

Without swizzling, the 4 core matrices would be stored across 4 SMEM rows resulting in bank conflicts when loading the 8x64B atom.

```
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 0
Row 1   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 1
Row 2   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 2
Row 3   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 3
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

No bank-conflict layout with swizzling, 

```
        +-----------------------------------+-----------------------------------+
        |               Tile 0              |               Tile 1              |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 0
Row 1   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | Core Matrix 1
Row 2   |   2    |   3    |   0    |   1    |   6    |   7    |   4    |   5    | Core Matrix 2
Row 3   |   3    |   2    |   1    |   0    |   7    |   6    |   5    |   4    | Core Matrix 3
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

### 5.4 Example D: 128B Swizzle Atom — 8×128B (8 Core Matrices)

**Configuration:** `Swizzle<3, 4, 3>`

- BBits=3: 8×8 tile of swizzle units
- MBase=4: 16-byte units
- SShift=3: 8 units per row

**Swizzle tile:** 8 rows × 8 units (128 bytes wide) — **exactly one SMEM row!**

**XOR values by row:** 0, 1, 2, 3, 4, 5, 6, 7 (all unique)

**Physical Layout:**

```
        +-----------------------------------------------------------------------+
        |                                 Tile 0                                |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | XOR 0
Row 1   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | XOR 1
Row 2   |   2    |   3    |   0    |   1    |   6    |   7    |   4    |   5    | XOR 2
Row 3   |   3    |   2    |   1    |   0    |   7    |   6    |   5    |   4    | XOR 3
Row 4   |   4    |   5    |   6    |   7    |   0    |   1    |   2    |   3    | XOR 4
Row 5   |   5    |   4    |   7    |   6    |   1    |   0    |   3    |   2    | XOR 5
Row 6   |   6    |   7    |   4    |   5    |   2    |   3    |   0    |   1    | XOR 6
Row 7   |   7    |   6    |   5    |   4    |   3    |   2    |   1    |   0    | XOR 7
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

**Column Access Pattern (reading logical Unit 0):**

- Row 0: Physical Unit 0 (Banks 0-3)
- Row 1: Physical Unit 1 (Banks 4-7)
- Row 2: Physical Unit 2 (Banks 8-11)
- Row 3: Physical Unit 3 (Banks 12-15)
- Row 4: Physical Unit 4 (Banks 16-19)
- Row 5: Physical Unit 5 (Banks 20-23)
- Row 6: Physical Unit 6 (Banks 24-27)
- Row 7: Physical Unit 7 (Banks 28-31)

**All 8 rows access different banks!**

**Tensor Core access Pattern for 8 core matrix wide swizzle atom (8x128B):**

No bank conflict. 

Without swizzling, the 8 core matrices would be stored across 8 SMEM rows resulting in bank conflicts when loading the 8x128B atom.

```
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 0
Row 1   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 1
Row 2   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 2
Row 3   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 3
Row 4   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 4
Row 5   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 5
Row 6   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 6
Row 7   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 7
        +--------+--------+--------+--------+--------+--------+--------+--------+
```
No bank-conflict layout with swizzling, 

```
        +-----------------------------------------------------------------------+
        |                                 Tile 0                                |
        +-----------------+-----------------+-----------------+-----------------+
        | Unit 0 | Unit 1 | Unit 2 | Unit 3 | Unit 4 | Unit 5 | Unit 6 | Unit 7 |
        +--------+--------+--------+--------+--------+--------+--------+--------+
Row 0   |   0    |   1    |   2    |   3    |   4    |   5    |   6    |   7    | Core Matrix 0
Row 1   |   1    |   0    |   3    |   2    |   5    |   4    |   7    |   6    | Core Matrix 1
Row 2   |   2    |   3    |   0    |   1    |   6    |   7    |   4    |   5    | Core Matrix 2
Row 3   |   3    |   2    |   1    |   0    |   7    |   6    |   5    |   4    | Core Matrix 3
Row 4   |   4    |   5    |   6    |   7    |   0    |   1    |   2    |   3    | Core Matrix 4
Row 5   |   5    |   4    |   7    |   6    |   1    |   0    |   3    |   2    | Core Matrix 5
Row 6   |   6    |   7    |   4    |   5    |   2    |   3    |   0    |   1    | Core Matrix 6
Row 7   |   7    |   6    |   5    |   4    |   3    |   2    |   1    |   0    | Core Matrix 7
        +--------+--------+--------+--------+--------+--------+--------+--------+
```

### 5.5 Example E: 128B Swizzle Atom with 32B Swizzle Unit — 8×128B

**Configuration:** `Swizzle<2, 5, 2>`

- BBits=2: 4×4 tile of swizzle units
- MBase=5: 32-byte units (8 banks each)
- SShift=2: 4 units per row

**Swizzle tile:** 4 rows × 4 units (128 bytes wide)

This configuration uses larger swizzle units (32B instead of 16B), useful when threads load 32 bytes at a time (e.g., 8×fp32 or 16×fp16).

**XOR values by row:**

- Row 0, 4: XOR with 0
- Row 1, 5: XOR with 1
- Row 2, 6: XOR with 2
- Row 3, 7: XOR with 3

**Physical Layout:**

```
        +-------------------------------------------------------------------+
        |                               Tile 0                              |
        +----------------+----------------+----------------+----------------+
        |  Unit 0 (32B)  |  Unit 1 (32B)  |  Unit 2 (32B)  |  Unit 3 (32B)  |
        +----------------+----------------+----------------+----------------+
Row 0   |       0        |       1        |       2        |       3        | XOR 0
Row 1   |       1        |       0        |       3        |       2        | XOR 1
Row 2   |       2        |       3        |       0        |       1        | XOR 2
Row 3   |       3        |       2        |       1        |       0        | XOR 3
        +----------------+----------------+----------------+----------------+
Row 4   |       0        |       1        |       2        |       3        | XOR 0
Row 5   |       1        |       0        |       3        |       2        | XOR 1
Row 6   |       2        |       3        |       0        |       1        | XOR 2
Row 7   |       3        |       2        |       1        |       0        | XOR 3
        +----------------+----------------+----------------+----------------+
```

**Bank Mapping (each 32B unit spans 8 banks):**

```
SMEM Banks:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
            +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
Row 0       |              L0               |              L1               |              L2               |              L3               |
Row 1       |              L1               |              L0               |              L3               |              L2               |
Row 2       |              L2               |              L3               |              L0               |              L1               |
Row 3       |              L3               |              L2               |              L1               |              L0               |
Row 4       |              L0               |              L1               |              L2               |              L3               |
...
```

## 6. Key Takeaways

1. **MBase** sets the granularity—choose based on your natural access width (16B for fp16×8, 32B for fp16×16, etc.)

2. **BBits** determines conflict reduction—larger BBits = more XOR values = fewer conflicts. For 8-row access, BBits=3 eliminates conflicts completely.

3. **SShift** connects to SMEM geometry—typically `7 - MBase` for 128-byte SMEM rows.

4. **Swizzle tiles are independent**—no coordination needed across tile boundaries. Each tile applies its own XOR pattern.

5. **The formula is simple:**
   ```
   physical_slot = (tile_index << BBits) | (logical_col_in_tile XOR row_mod)
   ```

## 7. Swizzle Visualizer

I "claude-coded" an HTML+JS based swizzle visualizer that let's you configure the swizzle unit, swizzle tile size and grid configuration and then visualize swizzled layouts in action.

<div class="swizzle-visualizer-inline">
    <style>
        .swizzle-visualizer-inline * {
            box-sizing: border-box;
        }

        .swizzle-visualizer-inline {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            width: 100%;
            max-width: none;
            margin: 0;
            padding: 20px;
            background: #000;
            color: #eee;
        }

        .swizzle-visualizer-inline h1 {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
        }

        .control-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 15px;
            background: #1a1a2e;
            border-radius: 8px;
        }

        .section-title {
            font-size: 12px;
            color: #00d4ff;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .control-row {
            display: flex;
            gap: 15px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .control-group label {
            font-weight: bold;
            margin-bottom: 5px;
            color: #00d4ff;
        }

        .control-group .description {
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
            text-align: center;
        }

        .control-group input, .control-group select {
            padding: 8px 12px;
            font-size: 16px;
            border: 2px solid #00d4ff;
            border-radius: 5px;
            background: #1a1a2e;
            color: #fff;
            width: 120px;
            text-align: center;
            -moz-appearance: textfield;
        }

        .control-group input::-webkit-outer-spin-button,
        .control-group input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .control-group input:disabled {
            border-color: #888;
            color: #888;
            cursor: not-allowed;
        }

        .computed-value {
            margin-top: 5px;
            font-size: 11px;
            color: #00d4ff;
            text-align: center;
        }

        .tile-width-info, .computed-info {
            margin-top: 10px;
            font-size: 12px;
            color: #00d4ff;
            text-align: center;
            width: 100%;
        }

        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #888;
            cursor: pointer;
        }

        .checkbox-label input[type="checkbox"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .bank-info {
            margin-top: 8px;
            font-size: 11px;
            color: #00d4ff;
            text-align: center;
        }

        .canonical-swizzle {
            width: 100%;
            margin-top: 10px;
            padding: 12px;
            text-align: center;
            font-family: 'Courier New', monospace;
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
        }

        .canonical-swizzle span {
            transition: color 0.15s ease;
        }

        .canonical-swizzle span.highlight {
            color: #4ade80;
        }

        .address-info {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
        }

        .address-info h3 {
            margin-top: 0;
            color: #00d4ff;
        }

        .address-hint {
            color: #888;
            font-size: 13px;
        }

        .address-details {
            display: none;
            margin-top: 15px;
        }

        .address-details.visible {
            display: block;
        }

        .address-row {
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .address-item {
            background: #1a1a2e;
            padding: 10px 15px;
            border-radius: 5px;
        }

        .address-label {
            font-size: 11px;
            color: #888;
            margin-bottom: 5px;
        }

        .address-value {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: #00d4ff;
        }

        .address-binary {
            font-family: 'Courier New', monospace;
            font-size: 13px;
            margin-top: 10px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 5px;
            line-height: 1.8;
        }

        .address-binary .zzz-bits {
            color: #ff6b6b;
            font-weight: bold;
        }

        .address-binary .yyy-bits {
            color: #4ecdc4;
            font-weight: bold;
        }

        .cell {
            cursor: pointer;
        }

        .cell:hover {
            opacity: 0.8;
        }

        .cell.selected {
            outline: 3px solid #000;
            outline-offset: -3px;
        }

        .constraint-warning {
            display: none;
            background: #ff6b6b;
            color: #000;
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }

        .stats {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
        }

        .stat-group {
            background: #16213e;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }

        .stat-group-title {
            font-size: 12px;
            color: #00d4ff;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #333;
        }

        .stat-group-items {
            display: flex;
            gap: 15px;
            justify-content: center;
        }

        .stat {
            padding: 5px 10px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: bold;
            color: #fff;
        }

        .stat-label {
            font-size: 11px;
            color: #888;
        }

        .tile-section {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        .tile-section h3 {
            color: #00d4ff;
            margin-top: 0;
            margin-bottom: 10px;
        }

        .tile-description {
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }

        .tile-visualization {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }

        .tile-container {
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
        }

        .tile-arrow {
            font-size: 32px;
            color: #00d4ff;
        }

        .grid-section {
            text-align: center;
            margin-bottom: 30px;
        }

        .grid-section h3 {
            color: #00d4ff;
            margin-bottom: 10px;
        }

        .grid-description {
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }

        .bank-section {
            text-align: center;
            margin-bottom: 30px;
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }

        .bank-section h3 {
            color: #00d4ff;
            margin-top: 0;
            margin-bottom: 10px;
        }

        .bank-description {
            color: #888;
            font-size: 13px;
            margin-bottom: 20px;
        }

        .bank-view {
            display: flex;
            flex-direction: column;
            gap: 2px;
            align-items: center;
        }

        .bank-row {
            display: flex;
            gap: 1px;
            align-items: center;
        }

        .bank-row-label {
            width: 60px;
            text-align: right;
            padding-right: 10px;
            font-size: 11px;
            color: #888;
        }

        .bank-cell {
            width: 20px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: bold;
            color: #000;
            border-radius: 2px;
        }

        .bank-cell.selected-start {
            border-left: 2px solid #000;
            border-top: 2px solid #000;
            border-bottom: 2px solid #000;
        }

        .bank-cell.selected-middle {
            border-top: 2px solid #000;
            border-bottom: 2px solid #000;
        }

        .bank-cell.selected-end {
            border-right: 2px solid #000;
            border-top: 2px solid #000;
            border-bottom: 2px solid #000;
        }

        .bank-cell.selected-single {
            outline: 2px solid #000;
            outline-offset: -2px;
        }

        .bank-header {
            display: flex;
            gap: 1px;
            margin-left: 70px;
            margin-bottom: 5px;
        }

        .bank-header-cell {
            width: 20px;
            text-align: center;
            font-size: 9px;
            color: #888;
        }

        .bank-tile-container {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }

        .bank-tile-box {
            border: 2px solid #000;
            border-radius: 4px;
            padding: 2px;
            background: #1a1a2e;
        }

        .bank-tile-label {
            margin-left: 10px;
            font-size: 11px;
            color: #00d4ff;
            white-space: nowrap;
        }

        .visualization {
            display: flex;
            gap: 40px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .grid-container {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
        }

        .grid-title {
            text-align: center;
            margin-bottom: 5px;
            font-weight: bold;
            color: #00d4ff;
        }

        .grid-subtitle {
            text-align: center;
            margin-bottom: 15px;
            font-size: 11px;
            color: #888;
        }

        .grid-wrapper {
            overflow-x: auto;
        }

        .grid {
            display: inline-block;
            border-collapse: collapse;
        }

        .grid-row {
            display: flex;
        }

        .cell {
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            border: 1px solid #333;
            transition: all 0.2s;
        }

        .cell.highlighted {
            outline: 3px solid #fff;
            outline-offset: -3px;
            z-index: 10;
            position: relative;
        }

        .row-label {
            width: 50px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: #888;
            background: #1a1a2e;
        }

        .col-labels {
            display: flex;
            margin-left: 50px;
            margin-bottom: 5px;
        }

        .col-label {
            width: 40px;
            text-align: center;
            font-size: 12px;
            color: #888;
        }

        .legend {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
        }

        .legend h3 {
            margin-top: 0;
            color: #00d4ff;
        }

        .legend-items {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 10px;
            background: #1a1a2e;
            border-radius: 5px;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }

        .explanation {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
            line-height: 1.6;
        }

        .explanation h3 {
            margin-top: 0;
            color: #00d4ff;
        }

        .explanation code {
            background: #1a1a2e;
            padding: 2px 6px;
            border-radius: 3px;
            color: #00d4ff;
        }

        .tile-separator-v {
            border-right: 4px solid #000 !important;
        }

        .tile-separator-h {
            border-bottom: 4px solid #000 !important;
        }

        .highlight-control {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .highlight-control input[type="checkbox"] {
            width: 20px;
            height: 20px;
        }

        .conflict-check {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
            display: none;
        }

        .conflict-check.visible {
            display: block;
        }

        .conflict-check h3 {
            margin-top: 0;
            color: #00d4ff;
        }

        .conflict-check .result {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }

        .conflict-check .slots {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }

        .conflict-check .slot {
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #1a1a2e;
            border-radius: 5px;
            font-weight: bold;
            font-size: 14px;
        }

        .conflict-check .verdict {
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
        }

        .conflict-check .verdict.success {
            background: #1fab89;
            color: #000;
        }

        .conflict-check .verdict.failure {
            background: #ff6b6b;
            color: #000;
        }

        .formula-box {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
        }

        .formula-box h3 {
            margin-top: 0;
            color: #00d4ff;
        }

        .formula {
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 14px;
            line-height: 1.6;
        }

        .formula .comment {
            color: #888;
        }
    </style>
    <h1>GPU Shared Memory Swizzle Visualizer</h1>
    <p class="subtitle">Visualize how <code>Swizzle&lt;BBits, MBase, SShift&gt;</code> remaps addresses to avoid bank conflicts</p>

    <div class="controls">
        <div class="control-section" style="display: flex; flex-direction: column; justify-content: center;">
            <div class="section-title">Presets</div>
            <div class="control-row" style="flex: 1; display: flex; align-items: center; justify-content: center;">
                <div class="control-group">
                    <select id="presets" style="width: 200px;">
                        <option value="">Custom</option>
                        <option value="0,4,3">No swizzle atom &lt;0,4,3&gt;</option>
                        <option value="1,4,3">32B swizzle atom &lt;1,4,3&gt;</option>
                        <option value="2,4,3">64B swizzle atom &lt;2,4,3&gt;</option>
                        <option value="3,4,3" selected>128B swizzle atom &lt;3,4,3&gt;</option>
                        <option value="2,5,2">128B atom, 32B unit &lt;2,5,2&gt;</option>
                    </select>
                </div>
            </div>
        </div>
        <div class="control-section">
            <div class="section-title">Swizzle Unit</div>
            <div class="control-row">
                <div class="control-group">
                    <label>MBase</label>
                    <div class="description">Unit size = 2^MBase bytes</div>
                    <input type="number" id="mbase" value="4" min="2" max="6">
                </div>
            </div>
            <div class="computed-info" id="unitSizeInfo"></div>
        </div>
        <div class="control-section">
            <div class="section-title">Swizzle Tile</div>
            <div class="control-row">
                <div class="control-group">
                    <label>BBits</label>
                    <div class="description">Tile = 2^BBits x 2^BBits units</div>
                    <input type="number" id="bbits" value="3" min="1" max="5">
                </div>
            </div>
            <div class="computed-info" id="tileWidthInfo"></div>
        </div>
        <div class="control-section">
            <div class="section-title">Grid Configuration</div>
            <div class="control-row">
                <div class="control-group">
                    <label>SShift</label>
                    <div class="description">Swz units/grid row = 2^SShift</div>
                    <input type="number" id="sshift" value="3" min="0" max="6">
                </div>
                <div class="control-group">
                    <label>Grid Rows</label>
                    <div class="description">Default = tile height</div>
                    <input type="number" id="displayRows" value="8" min="1" max="32">
                </div>
            </div>
            <div class="control-row" style="margin-top: 10px;">
                <label class="checkbox-label">
                    <input type="checkbox" id="matchBankWidth" checked="checked">
                    <span>Match number of SMEM banks (32 DWords / 128B)</span>
                </label>
            </div>
            <div class="bank-info" id="bankInfo"></div>
        </div>
        <div class="control-section">
            <div class="section-title">Inspect</div>
            <div class="control-row">
                <div class="control-group">
                    <label>Highlight Column</label>
                    <div class="description">Trace logical column</div>
                    <select id="highlightCol">
                        <option value="-1">None</option>
                    </select>
                </div>
            </div>
        </div>
        <div class="canonical-swizzle" id="canonicalSwizzle"></div>
    </div>

    <div class="constraint-warning" id="constraint-warning"></div>
    <div class="stats" id="stats"></div>

    <div class="legend" id="legend"></div>

    <div class="tile-section">
        <h3>Single Swizzle Tile</h3>
        <p class="tile-description">This is one swizzle tile. The grid below is composed of these tiles.</p>
        <div class="tile-visualization">
            <div class="tile-container">
                <div class="grid-title">Tile WITHOUT Swizzle</div>
                <div class="grid-wrapper" id="tileWrapperNoSwizzle"></div>
            </div>
            <div class="tile-arrow">-></div>
            <div class="tile-container">
                <div class="grid-title">Tile WITH Swizzle</div>
                <div class="grid-wrapper" id="tileWrapper"></div>
            </div>
        </div>
    </div>

    <div class="grid-section">
        <h3>Full Grid Layout</h3>
        <p class="grid-description">The grid is composed of swizzle tiles. Tile boundaries are marked with thick black lines.</p>
        <div class="visualization">
            <div class="grid-container">
                <div class="grid-title">Grid WITHOUT Swizzle</div>
                <div class="grid-subtitle">Color & number = logical column</div>
                <div class="grid-wrapper" id="gridWrapperNoSwizzle"></div>
            </div>
            <div class="grid-container">
                <div class="grid-title">Grid WITH Swizzle</div>
                <div class="grid-subtitle">Color & number = logical column stored at each physical slot</div>
                <div class="grid-wrapper" id="gridWrapper"></div>
            </div>
        </div>
    </div>

    <div class="address-info" id="addressInfo">
        <h3>Address Inspector</h3>
        <p class="address-hint">Click any cell in the grids above or SMEM bank organization below to see its address details.</p>
        <div class="address-details" id="addressDetails"></div>
    </div>

    <div class="bank-section">
        <h3>SMEM Bank Organization</h3>
        <p class="bank-description">Bank-level view of SMEM with standard row-major layout and swizzle applied. Each grid row is stored consecutively in memory. Numbers show logical column stored at each bank position.</p>
        <div class="bank-view" id="bankView"></div>
    </div>

    <div class="conflict-check" id="conflictCheck"></div>

    <div class="formula-box">
        <h3>Swizzle Formula</h3>
        <div class="formula">
            <span class="comment">// For swz unit at logical position (row, col):</span><br><br>
            mask = (1 &lt;&lt; BBits) - 1<br>
            col_in_tile = col &amp; mask <span class="comment">// col mod 2^BBits</span><br>
            row_mod = row &amp; mask <span class="comment">// row mod 2^BBits</span><br>
            swizzled_slot = col_in_tile ^ row_mod<br><br>
            <span class="comment">// The swizzled slot determines which bank group is accessed.</span><br>
            <span class="comment">// Looking down any column: all row_mod values are different,</span><br>
            <span class="comment">// so all swizzled_slot values are different -> no bank conflicts!</span>
        </div>
    </div>

    <div class="explanation">
        <h3>How to Read This Visualization</h3>
        <p>
            <strong>Swizzle Tile:</strong> The fundamental swizzle unit (2^BBits x 2^BBits swz units). The swizzle pattern is defined within a tile.
        </p>
        <p>
            <strong>Grid:</strong> The full shared memory layout, composed of one or more swizzle tiles. Tile boundaries are marked with thick black lines.
        </p>
        <p>
            <strong>Colors:</strong> Each color represents a <em>logical column</em>. In the swizzled view, colors show which logical column's data is stored at each physical slot.
        </p>
        <p>
            <strong>WITHOUT Swizzle:</strong> Looking down any column, all cells have the same color -> bank conflict when reading column-major.
        </p>
        <p>
            <strong>WITH Swizzle:</strong> Looking down any column, colors are shuffled -> no bank conflicts, data spread across different logical columns.
        </p>
        <p>
            <strong>To verify:</strong> Use "Highlight Slot" to see which logical columns are accessed when reading a physical slot across rows.
        </p>
    </div>

    <script>
        // Generate colors using golden angle for maximum visual separation
        // Golden angle (~137.5°) ensures adjacent indices have very different hues
        function getColor(colIndex, totalCols) {
            const goldenAngle = 137.508;  // Golden angle in degrees
            const hue = (colIndex * goldenAngle) % 360;
            // Vary saturation and lightness slightly for additional distinction
            const saturation = 60 + (colIndex % 3) * 10;  // 60%, 70%, or 80%
            const lightness = 55 + (colIndex % 2) * 15;   // 55% or 70%
            return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
        }

        // For tile-only views, use the same color scheme
        function getTileColor(colInTile, tileCols) {
            return getColor(colInTile, tileCols);
        }

        function swizzle(row, col, bbits, sshift) {
            const mask = (1 << bbits) - 1;
            const tileIdx = col >> bbits;
            const colInTile = col & mask;
            const rowMod = row & mask;
            const swizzledColInTile = colInTile ^ rowMod;
            return swizzledColInTile;
        }

        function render() {
            const bbits = parseInt(document.getElementById('bbits').value);
            const mbase = parseInt(document.getElementById('mbase').value);
            const sshift = parseInt(document.getElementById('sshift').value);
            const displayRows = parseInt(document.getElementById('displayRows').value);
            const highlightCol = parseInt(document.getElementById('highlightCol').value);

            const constraintEl = document.getElementById('constraint-warning');

            // Validate MBase >= 2 (swizzle unit must be at least one bank = 4 bytes)
            if (mbase < 2) {
                constraintEl.style.display = 'block';
                constraintEl.innerHTML = `
                    <strong>Invalid Configuration</strong><br><br>
                    <strong>Problem:</strong> Swizzle unit size (${1 << mbase} bytes) is smaller than one SMEM bank (4 bytes).<br><br>
                    <strong>Why:</strong> Swizzle unit must be at least one bank (4 bytes / 1 DWord) in size.<br><br>
                    <strong>Fix:</strong> Set MBase >= 2 (unit size >= 4 bytes).
                `;

                // Clear all visualizations
                document.getElementById('stats').innerHTML = '';
                document.getElementById('gridWrapper').innerHTML = '';
                document.getElementById('gridWrapperNoSwizzle').innerHTML = '';
                document.getElementById('tileWrapper').innerHTML = '';
                document.getElementById('tileWrapperNoSwizzle').innerHTML = '';
                document.getElementById('legend').innerHTML = '';
                document.getElementById('bankView').innerHTML = '';
                document.getElementById('conflictCheck').classList.remove('visible');
                return;
            }

            // Validate sshift >= bbits
            if (sshift < bbits) {
                const tileWidth = 1 << bbits;
                const unitsPerRow = 1 << sshift;

                constraintEl.style.display = 'block';
                constraintEl.innerHTML = `
                    <strong>Invalid Configuration</strong><br><br>
                    <strong>Problem:</strong> Row width (${unitsPerRow} swz units) is smaller than tile width (${tileWidth} swz units).<br><br>
                    <strong>Why:</strong> A ${tileWidth}x${tileWidth} swizzle tile cannot fit in a row that's only ${unitsPerRow} swz units wide.<br><br>
                    <strong>Fix:</strong> Either reduce BBits to <=${sshift} (smaller tile), or increase SShift to >=${bbits} (wider rows).
                `;

                document.getElementById('stats').innerHTML = '';
                document.getElementById('gridWrapper').innerHTML = '';
                document.getElementById('gridWrapperNoSwizzle').innerHTML = '';
                document.getElementById('tileWrapper').innerHTML = '';
                document.getElementById('tileWrapperNoSwizzle').innerHTML = '';
                document.getElementById('legend').innerHTML = '';
                document.getElementById('bankView').innerHTML = '';
                document.getElementById('conflictCheck').classList.remove('visible');
                return;
            }

            constraintEl.style.display = 'none';

            const tileRows = 1 << bbits;
            const tileCols = 1 << bbits;
            const unitsPerRow = 1 << sshift;
            const unitSize = 1 << mbase;
            const tilesPerRow = 1 << (sshift - bbits);
            const rowWidth = unitsPerRow * unitSize;
            const tileSize = tileCols * tileRows * unitSize;
            const numTileRows = Math.ceil(displayRows / tileRows);
            const totalColors = numTileRows * unitsPerRow;  // unique color per column per tile

            currentConfig = { bbits, mbase, sshift, numTileRows, tileRows, tileCols, unitsPerRow, tilesPerRow };

            const numBanks = unitSize / 4;
            document.getElementById('unitSizeInfo').textContent = `= ${unitSize} bytes (${numBanks} banks)`;

            const tileWidthBytes = tileCols * unitSize;
            document.getElementById('tileWidthInfo').innerHTML = `Tile width = ${tileWidthBytes}B<br>Tile height = ${tileRows} rows`;

            document.getElementById('canonicalSwizzle').innerHTML = `Swizzle&lt;<span id="swz-bbits">${bbits}</span>, <span id="swz-mbase">${mbase}</span>, <span id="swz-sshift">${sshift}</span>&gt;`;

            const activeEl = document.activeElement;
            if (activeEl && activeEl.id === 'bbits') {
                document.getElementById('swz-bbits').classList.add('highlight');
            } else if (activeEl && activeEl.id === 'mbase') {
                document.getElementById('swz-mbase').classList.add('highlight');
            } else if (activeEl && activeEl.id === 'sshift') {
                document.getElementById('swz-sshift').classList.add('highlight');
            }

            document.getElementById('stats').innerHTML = `
                <div class="stat-group">
                    <div class="stat-group-title">Swizzle Unit</div>
                    <div class="stat-group-items">
                        <div class="stat">
                            <div class="stat-value">${unitSize}B</div>
                            <div class="stat-label">Size</div>
                        </div>
                    </div>
                </div>
                <div class="stat-group">
                    <div class="stat-group-title">Tile</div>
                    <div class="stat-group-items">
                        <div class="stat">
                            <div class="stat-value">${tileRows}x${tileCols}</div>
                            <div class="stat-label">Rows x Units</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${tileSize}B</div>
                            <div class="stat-label">Size</div>
                        </div>
                    </div>
                </div>
                <div class="stat-group">
                    <div class="stat-group-title">Grid</div>
                    <div class="stat-group-items">
                        <div class="stat">
                            <div class="stat-value">${displayRows}x${unitsPerRow}</div>
                            <div class="stat-label">Rows x Units</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${tilesPerRow}</div>
                            <div class="stat-label">Tiles/Row</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${rowWidth}B</div>
                            <div class="stat-label">Row Width</div>
                        </div>
                    </div>
                </div>
            `;

            const highlightSelect = document.getElementById('highlightCol');
            const currentHighlight = highlightSelect.value;
            highlightSelect.innerHTML = '<option value="-1">None</option>';
            for (let c = 0; c < tileCols; c++) {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = `Col ${c}`;
                opt.style.color = getColor(c, totalColors);
                highlightSelect.appendChild(opt);
            }
            highlightSelect.value = currentHighlight < tileCols ? currentHighlight : -1;

            // Build swizzled grid
            let html = '';
            html += '<div class="col-labels">';
            for (let c = 0; c < unitsPerRow; c++) {
                html += `<div class="col-label">${c}</div>`;
            }
            html += '</div>';

            const mask = (1 << bbits) - 1;
            for (let r = 0; r < displayRows; r++) {
                html += '<div class="grid-row">';
                html += `<div class="row-label">Row ${r}</div>`;

                for (let physicalSlot = 0; physicalSlot < unitsPerRow; physicalSlot++) {
                    const physicalSlotInTile = physicalSlot & mask;
                    const rowMod = r & mask;
                    const logicalColInTile = physicalSlotInTile ^ rowMod;

                    const tileColIdx = physicalSlot >> bbits;
                    const tileRowIdx = r >> bbits;
                    const fullLogicalCol = (tileColIdx << bbits) | logicalColInTile;
                    const colorIndex = tileRowIdx * unitsPerRow + fullLogicalCol;
                    const color = getColor(colorIndex, totalColors);
                    const textColor = '#000';

                    let classes = 'cell';
                    if ((physicalSlot + 1) % tileCols === 0 && physicalSlot < unitsPerRow - 1) {
                        classes += ' tile-separator-v';
                    }
                    if ((r + 1) % tileRows === 0 && r < displayRows - 1) {
                        classes += ' tile-separator-h';
                    }
                    const selectedLogicalCol = parseInt(highlightSelect.value);
                    if (selectedLogicalCol >= 0 && logicalColInTile === selectedLogicalCol) {
                        classes += ' highlighted';
                    }
                    html += `<div class="${classes}" style="background: ${color}; color: ${textColor};"
                             data-row="${r}" data-physical-col="${physicalSlot}" data-logical-col="${fullLogicalCol}"
                             onclick="showAddressDetails(${r}, ${fullLogicalCol}, true)"
                             title="Physical slot ${physicalSlot} <- Logical col ${logicalColInTile}">${logicalColInTile}</div>`;
                }

                html += '</div>';
            }

            document.getElementById('gridWrapper').innerHTML = html;

            // Build non-swizzled grid
            let htmlNoSwizzle = '';
            htmlNoSwizzle += '<div class="col-labels">';
            for (let c = 0; c < unitsPerRow; c++) {
                htmlNoSwizzle += `<div class="col-label">${c}</div>`;
            }
            htmlNoSwizzle += '</div>';

            for (let r = 0; r < displayRows; r++) {
                htmlNoSwizzle += '<div class="grid-row">';
                htmlNoSwizzle += `<div class="row-label">Row ${r}</div>`;

                for (let c = 0; c < unitsPerRow; c++) {
                    const colInTile = c & ((1 << bbits) - 1);
                    const tileRowIdx = r >> bbits;
                    const colorIndex = tileRowIdx * unitsPerRow + c;
                    const color = getColor(colorIndex, totalColors);
                    const textColor = '#000';

                    let classes = 'cell';
                    if ((c + 1) % tileCols === 0 && c < unitsPerRow - 1) {
                        classes += ' tile-separator-v';
                    }
                    if ((r + 1) % tileRows === 0 && r < displayRows - 1) {
                        classes += ' tile-separator-h';
                    }
                    const selectedLogicalCol = parseInt(highlightSelect.value);
                    if (selectedLogicalCol >= 0 && colInTile === selectedLogicalCol) {
                        classes += ' highlighted';
                    }

                    htmlNoSwizzle += `<div class="${classes}" style="background: ${color}; color: ${textColor};"
                             data-row="${r}" data-logical-col="${c}" data-physical-col="${c}"
                             onclick="showAddressDetails(${r}, ${c}, false)"
                             title="Row ${r}, Col ${c} -> Logical col ${colInTile}">${colInTile}</div>`;
                }

                htmlNoSwizzle += '</div>';
            }

            document.getElementById('gridWrapperNoSwizzle').innerHTML = htmlNoSwizzle;

            // Build tile visualizations
            let tileHtmlNoSwizzle = '';
            tileHtmlNoSwizzle += '<div class="col-labels">';
            for (let c = 0; c < tileCols; c++) {
                tileHtmlNoSwizzle += `<div class="col-label">${c}</div>`;
            }
            tileHtmlNoSwizzle += '</div>';

            for (let r = 0; r < tileRows; r++) {
                tileHtmlNoSwizzle += '<div class="grid-row">';
                tileHtmlNoSwizzle += `<div class="row-label">Row ${r}</div>`;
                for (let c = 0; c < tileCols; c++) {
                    const color = getColor(c, totalColors);
                    tileHtmlNoSwizzle += `<div class="cell" style="background: ${color}; color: #000;"
                                          data-row="${r}" data-logical-col="${c}"
                                          onclick="showAddressDetails(${r}, ${c}, false)">${c}</div>`;
                }
                tileHtmlNoSwizzle += '</div>';
            }
            document.getElementById('tileWrapperNoSwizzle').innerHTML = tileHtmlNoSwizzle;

            let tileHtml = '';
            tileHtml += '<div class="col-labels">';
            for (let c = 0; c < tileCols; c++) {
                tileHtml += `<div class="col-label">${c}</div>`;
            }
            tileHtml += '</div>';

            for (let r = 0; r < tileRows; r++) {
                tileHtml += '<div class="grid-row">';
                tileHtml += `<div class="row-label">Row ${r}</div>`;
                for (let physicalSlot = 0; physicalSlot < tileCols; physicalSlot++) {
                    const rowMod = r & mask;
                    const logicalCol = physicalSlot ^ rowMod;
                    const color = getColor(logicalCol, totalColors);
                    tileHtml += `<div class="cell" style="background: ${color}; color: #000;"
                                 data-row="${r}" data-logical-col="${logicalCol}"
                                 onclick="showAddressDetails(${r}, ${logicalCol}, true)"
                                 title="Physical slot ${physicalSlot} <- Logical col ${logicalCol}">${logicalCol}</div>`;
                }
                tileHtml += '</div>';
            }
            document.getElementById('tileWrapper').innerHTML = tileHtml;

            // Build legend - show first tile's columns only (colors vary by tile)
            let legendHtml = '<h3>Logical Column Colors (first tile)</h3><div class="legend-items">';
            for (let c = 0; c < tileCols; c++) {
                const color = getColor(c, totalColors);
                legendHtml += `
                    <div class="legend-item">
                        <div class="legend-color" style="background: ${color};"></div>
                        <span>Col ${c}</span>
                    </div>
                `;
            }
            legendHtml += '</div>';
            document.getElementById('legend').innerHTML = legendHtml;

            // Build SMEM bank view - standard row-major layout with swizzle applied
            const totalBanks = 32;
            const banksPerUnit = unitSize / 4;
            const smemRowBytesCalc = 128;  // 32 banks * 4 bytes
            const unitsPerSmemRow = smemRowBytesCalc / unitSize;

            let bankHtml = '';

            // Calculate how many SMEM rows we need per grid row
            const smemRowsPerGridRow = Math.ceil(unitsPerRow / unitsPerSmemRow);

            // Header row with bank numbers
            bankHtml += '<div class="bank-header">';
            for (let b = 0; b < totalBanks; b++) {
                bankHtml += `<div class="bank-header-cell">${b}</div>`;
            }
            bankHtml += '</div>';

            // Iterate row by row (standard row-major layout)
            for (let gridRow = 0; gridRow < displayRows; gridRow++) {
                const tileRowIdx = gridRow >> bbits;
                const rowInTile = gridRow & mask;

                // For each grid row, show SMEM row(s) it occupies
                for (let subRow = 0; subRow < smemRowsPerGridRow; subRow++) {
                    const smemRowIdx = gridRow * smemRowsPerGridRow + subRow;

                    bankHtml += '<div class="bank-row">';
                    bankHtml += `<div class="bank-row-label">Row ${gridRow}${smemRowsPerGridRow > 1 ? '.' + subRow : ''}</div>`;

                    for (let bank = 0; bank < totalBanks; bank++) {
                        // Calculate which physical slot this bank corresponds to
                        const unitInSmemRow = Math.floor(bank / banksPerUnit);
                        const physicalSlot = subRow * unitsPerSmemRow + unitInSmemRow;

                        if (physicalSlot >= unitsPerRow) {
                            bankHtml += `<div class="bank-cell" style="background: #222; opacity: 0.3;"></div>`;
                            continue;
                        }

                        // Inverse swizzle: physical slot -> logical column
                        const physicalSlotInTile = physicalSlot & mask;
                        const rowMod = gridRow & mask;
                        const logicalColInTile = physicalSlotInTile ^ rowMod;
                        const tileColIdx = physicalSlot >> bbits;
                        const logicalCol = (tileColIdx << bbits) | logicalColInTile;

                        const bankOffsetInUnit = bank % banksPerUnit;
                        const dwordOffset = bankOffsetInUnit * 4;

                        const colorIndex = tileRowIdx * unitsPerRow + logicalCol;
                        const color = getColor(colorIndex, totalColors);
                        bankHtml += `<div class="bank-cell" style="background: ${color}; cursor: pointer;"
                                     data-smem-row="${smemRowIdx}" data-bank="${bank}" data-grid-row="${gridRow}"
                                     onclick="showAddressDetails(${gridRow}, ${logicalCol}, true, ${dwordOffset}, ${smemRowIdx}, ${bank})"
                                     title="Row ${gridRow}, Col ${logicalCol}">${logicalColInTile}</div>`;
                    }
                    bankHtml += '</div>';
                }
            }

            document.getElementById('bankView').innerHTML = bankHtml;

            // Update conflict check section
            const conflictCheckEl = document.getElementById('conflictCheck');
            const selectedCol = parseInt(highlightSelect.value);

            if (selectedCol >= 0) {
                conflictCheckEl.classList.add('visible');

                const tileHeight = 1 << bbits;
                const rowsInOneTile = Math.min(displayRows, tileHeight);
                const logicalCol = selectedCol;
                const selectedColColor = getColor(selectedCol, totalColors);

                const physicalSlotsNoSwizzle = [];
                for (let r = 0; r < rowsInOneTile; r++) {
                    physicalSlotsNoSwizzle.push(logicalCol);
                }

                const physicalSlotsSwizzle = [];
                for (let r = 0; r < rowsInOneTile; r++) {
                    const rowMod = r & ((1 << bbits) - 1);
                    physicalSlotsSwizzle.push(logicalCol ^ rowMod);
                }
                const uniqueSwizzle = new Set(physicalSlotsSwizzle);
                const allDifferentSwizzle = uniqueSwizzle.size === physicalSlotsSwizzle.length;

                let conflictHtml = `
                    <h3>Column ${logicalCol} - Physical slot mapping</h3>
                    <p style="color: #888; margin-bottom: 15px;">Tracing column ${logicalCol} across ${rowsInOneTile} rows (one tile height):</p>

                    <div class="result" style="margin-bottom: 15px;">
                        <div style="min-width: 120px;"><strong>WITHOUT swizzle:</strong></div>
                        <div class="slots">
                            ${physicalSlotsNoSwizzle.map((slot, i) => `<div class="slot" style="background: ${selectedColColor}; color: #000;">
                                <span title="Row ${i}">Slot ${slot}</span>
                            </div>`).join('')}
                        </div>
                        <div class="verdict failure">
                            ALL SAME SLOT -> ${rowsInOneTile}-WAY BANK CONFLICT!
                        </div>
                    </div>

                    <div class="result">
                        <div style="min-width: 120px;"><strong>WITH swizzle:</strong></div>
                        <div class="slots">
                            ${physicalSlotsSwizzle.map((slot, i) => `<div class="slot" style="background: ${selectedColColor}; color: #000;">
                                <span title="Row ${i}">Slot ${slot}</span>
                            </div>`).join('')}
                        </div>
                        <div class="verdict ${allDifferentSwizzle ? 'success' : 'failure'}">
                            ${allDifferentSwizzle ? 'ALL DIFFERENT SLOTS -> No bank conflicts!' : 'Some duplicates'}
                        </div>
                    </div>
                `;

                conflictCheckEl.innerHTML = conflictHtml;
            } else {
                conflictCheckEl.classList.remove('visible');
            }
        }

        let currentConfig = {};

        function showAddressDetails(row, logicalCol, isSwizzledGrid, dwordOffset = 0, smemRow = null, bank = null) {
            const { bbits, mbase, sshift, numTileRows, tileRows, tileCols, unitsPerRow } = currentConfig;
            const mask = (1 << bbits) - 1;
            const unitSize = 1 << mbase;
            const rowStride = unitsPerRow * unitSize;

            // Calculate tile and position within tile
            const tileColIdx = logicalCol >> bbits;
            const tileRowIdx = row >> bbits;
            const logicalColInTile = logicalCol & mask;
            const rowMod = row & mask;

            // Original address (without swizzle): standard row-major layout
            const originalAddr = row * rowStride + logicalCol * unitSize + dwordOffset;

            // Swizzled address: apply swizzle to column within tile
            const physicalSlotInTile = logicalColInTile ^ rowMod;
            const physicalCol = (tileColIdx << bbits) | physicalSlotInTile;
            const swizzledAddr = row * rowStride + physicalCol * unitSize + dwordOffset;

            function formatBinary(addr, highlightZZZ, highlightYYY) {
                const totalBits = Math.max(12, mbase + sshift + bbits + 4);
                let binary = addr.toString(2).padStart(totalBits, '0');
                let result = '';

                for (let i = 0; i < binary.length; i++) {
                    const bitPos = binary.length - 1 - i;
                    const bit = binary[i];

                    if (highlightYYY && bitPos >= mbase + sshift && bitPos < mbase + sshift + bbits) {
                        result += `<span class="yyy-bits">${bit}</span>`;
                    } else if (highlightZZZ && bitPos >= mbase && bitPos < mbase + bbits) {
                        result += `<span class="zzz-bits">${bit}</span>`;
                    } else {
                        result += bit;
                    }

                    if (bitPos > 0 && bitPos % 4 === 0) {
                        result += ' ';
                    }
                }
                return result;
            }

            // Calculate physical bank ID based on swizzled address
            const smemRowBytes = 32 * 4;  // 32 banks * 4 bytes
            const physicalBankId = Math.floor((swizzledAddr % smemRowBytes) / 4);

            const showByteOffset = typeof smemRow === 'number' || dwordOffset > 0;
            const dwordOffsetHtml = showByteOffset ? `
                    <div class="address-item">
                        <div class="address-label">Byte Offset</div>
                        <div class="address-value">+${dwordOffset} bytes (Bank ${physicalBankId})</div>
                    </div>` : '';

            const detailsEl = document.getElementById('addressDetails');
            detailsEl.classList.add('visible');
            detailsEl.innerHTML = `
                <div class="address-row">
                    <div class="address-item">
                        <div class="address-label">Grid Position</div>
                        <div class="address-value">Row ${row}, Col ${logicalCol}</div>
                    </div>
                    <div class="address-item">
                        <div class="address-label">Logical Column (in tile)</div>
                        <div class="address-value">${logicalColInTile}</div>
                    </div>
                    <div class="address-item">
                        <div class="address-label">Row mod 2^BBits</div>
                        <div class="address-value">${rowMod}</div>
                    </div>
                    <div class="address-item">
                        <div class="address-label">Physical Slot (in tile)</div>
                        <div class="address-value">${logicalColInTile} XOR ${rowMod} = ${physicalSlotInTile}</div>
                    </div>
                    ${dwordOffsetHtml}
                </div>
                <div class="address-row">
                    <div class="address-item">
                        <div class="address-label">Original Address (no swizzle)</div>
                        <div class="address-value">${originalAddr} (0x${originalAddr.toString(16).toUpperCase()})</div>
                    </div>
                    <div class="address-item">
                        <div class="address-label">Swizzled Address</div>
                        <div class="address-value">${swizzledAddr} (0x${swizzledAddr.toString(16).toUpperCase()})</div>
                    </div>
                </div>
                <div class="address-binary">
                    <strong>Binary breakdown:</strong><br>
                    <span style="color: #888;">Original:  </span>${formatBinary(originalAddr, true, true)}<br>
                    <span style="color: #888;">Swizzled:  </span>${formatBinary(swizzledAddr, true, true)}<br>
                    <br>
                    <span class="zzz-bits">ZZZ bits</span> (positions ${mbase}-${mbase + bbits - 1}): modified by swizzle<br>
                    <span class="yyy-bits">YYY bits</span> (positions ${mbase + sshift}-${mbase + sshift + bbits - 1}): XOR source (row bits)
                </div>
            `;

            // Highlight cells
            document.querySelectorAll('.cell.selected').forEach(el => el.classList.remove('selected'));
            document.querySelectorAll('.bank-cell.selected-start, .bank-cell.selected-middle, .bank-cell.selected-end, .bank-cell.selected-single').forEach(el => {
                el.classList.remove('selected-start', 'selected-middle', 'selected-end', 'selected-single');
            });

            document.querySelectorAll(`[data-row="${row}"][data-logical-col="${logicalCol}"]`).forEach(el => {
                el.classList.add('selected');
            });

            // Highlight corresponding bank cells in SMEM organization
            // Standard row-major layout with swizzle applied
            const totalBanksHighlight = 32;
            const banksPerUnitHighlight = unitSize / 4;
            const bytesPerSmemRowHighlight = 128;
            const unitsPerSmemRowHighlight = bytesPerSmemRowHighlight / unitSize;
            const smemRowsPerGridRow = Math.ceil(unitsPerRow / unitsPerSmemRowHighlight);

            // physicalCol is already calculated above
            const subRow = Math.floor(physicalCol / unitsPerSmemRowHighlight);
            const posInSubRow = physicalCol % unitsPerSmemRowHighlight;
            const smemRowHighlight = row * smemRowsPerGridRow + subRow;
            const startBankCol = posInSubRow * banksPerUnitHighlight;

            if (banksPerUnitHighlight === 1) {
                document.querySelectorAll(`[data-smem-row="${smemRowHighlight}"][data-bank="${startBankCol}"]`).forEach(el => {
                    el.classList.add('selected-single');
                });
            } else {
                for (let i = 0; i < banksPerUnitHighlight; i++) {
                    const bankCol = startBankCol + i;
                    let className;
                    if (i === 0) {
                        className = 'selected-start';
                    } else if (i === banksPerUnitHighlight - 1) {
                        className = 'selected-end';
                    } else {
                        className = 'selected-middle';
                    }
                    document.querySelectorAll(`[data-smem-row="${smemRowHighlight}"][data-bank="${bankCol}"]`).forEach(el => {
                        el.classList.add(className);
                    });
                }
            }
        }

        function calcBankMatchSShift(mbase) {
            return Math.max(0, 7 - mbase);
        }

        function updateBankMatch() {
            const matchCheckbox = document.getElementById('matchBankWidth');
            const mbase = parseInt(document.getElementById('mbase').value);
            const sshiftInput = document.getElementById('sshift');
            const bankInfo = document.getElementById('bankInfo');

            if (matchCheckbox.checked) {
                const newSShift = calcBankMatchSShift(mbase);
                sshiftInput.value = newSShift;
                sshiftInput.disabled = true;
                const unitSize = 1 << mbase;
                const unitsPerRow = 1 << newSShift;
                const rowBytes = unitSize * unitsPerRow;
                bankInfo.innerHTML = `Row = ${unitsPerRow} swz units x ${unitSize}B = ${rowBytes}B (32 banks x 4B)`;
            } else {
                sshiftInput.disabled = false;
                bankInfo.innerHTML = '';
            }
        }

        // Event listeners
        document.getElementById('bbits').addEventListener('input', function() {
            const newBbits = parseInt(this.value);
            const tileSize = 1 << newBbits;
            document.getElementById('displayRows').value = tileSize;
            render();
        });

        document.getElementById('mbase').addEventListener('input', function() {
            updateBankMatch();
            render();
        });

        document.getElementById('sshift').addEventListener('input', function() {
            document.getElementById('matchBankWidth').checked = false;
            updateBankMatch();
            render();
        });

        document.getElementById('matchBankWidth').addEventListener('change', function() {
            updateBankMatch();
            render();
        });

        document.getElementById('displayRows').addEventListener('input', render);
        document.getElementById('highlightCol').addEventListener('change', render);

        document.getElementById('presets').addEventListener('change', function() {
            const value = this.value;
            if (value) {
                const [bbits, mbase, sshift] = value.split(',').map(Number);
                document.getElementById('bbits').value = bbits;
                document.getElementById('mbase').value = mbase;
                document.getElementById('sshift').value = sshift;
                document.getElementById('displayRows').value = 8;
                document.getElementById('matchBankWidth').checked = false;
                updateBankMatch();
                render();
            }
        });

        function resetPresetToCustom() {
            document.getElementById('presets').value = '';
        }
        document.getElementById('bbits').addEventListener('input', resetPresetToCustom);
        document.getElementById('mbase').addEventListener('input', resetPresetToCustom);
        document.getElementById('sshift').addEventListener('input', resetPresetToCustom);

        function highlightSwizzleParam(paramId) {
            document.querySelectorAll('.canonical-swizzle span').forEach(el => el.classList.remove('highlight'));
            const el = document.getElementById(paramId);
            if (el) el.classList.add('highlight');
        }

        function clearSwizzleHighlight() {
            document.querySelectorAll('.canonical-swizzle span').forEach(el => el.classList.remove('highlight'));
        }

        document.getElementById('bbits').addEventListener('focus', () => highlightSwizzleParam('swz-bbits'));
        document.getElementById('bbits').addEventListener('blur', clearSwizzleHighlight);
        document.getElementById('mbase').addEventListener('focus', () => highlightSwizzleParam('swz-mbase'));
        document.getElementById('mbase').addEventListener('blur', clearSwizzleHighlight);
        document.getElementById('sshift').addEventListener('focus', () => highlightSwizzleParam('swz-sshift'));
        document.getElementById('sshift').addEventListener('blur', clearSwizzleHighlight);

        function init() {
            updateBankMatch();
            render();
            showAddressDetails(0, 0, true, 0, 0, 0);
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }
    </script>
</div>

## Conclusion

I hope this post helped you decode the canonical swizzle notation in CuTe. Please feel free to reach out if you spot any inaccuracies.

## Find me

[X](https://x.com/vjkrf1)
[GitHub](https://github.com/vj-krish)
[LinkedIn](https://www.linkedin.com/in/vjkrish3)
