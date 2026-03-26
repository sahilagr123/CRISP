"""Generate the CRISP pipeline diagram for the paper.

Grid-based layout with only 90° arrows, no overlaps.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 16))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors
C_COACH = '#F5E6CC'
C_PLAYER = '#DCE9F5'
C_VERIFY = '#E8E8E8'
C_DISCUSS = '#E8DDF5'
C_REWARD_P = '#D5F0D5'
C_REWARD_C = '#F5E8D0'
C_UPDATE_P = '#43A047'
C_UPDATE_C = '#FB8C00'
C_DECISION = '#FFF8DC'
C_SKIP = '#F0F0F0'
C_BORDER = '#666666'
C_TEXT = '#222222'


def box(x, y, w, h, color, lines, border_color=C_BORDER):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=border_color, linewidth=1.3)
    ax.add_patch(rect)
    n = len(lines)
    gap = h / (n + 1)
    for i, (text, fs, wt, col) in enumerate(lines):
        ax.text(x + w / 2, y + h - gap * (i + 1), text,
                ha='center', va='center', fontsize=fs, fontweight=wt, color=col)


def arr(x1, y1, x2, y2, color='#444444', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=3, shrinkB=3))


def line(x1, y1, x2, y2, color='#444444', lw=1.5):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle='round')


# ================================================================
# Layout
# ================================================================
# Coach box: x=0.8..4.0, centered at y=13.5
# Alice box: x=5.5..10.0, y=14.0
# Bob box:   x=5.5..10.0, y=12.6
# Main flow spine: x=7.75 (center of player boxes)
# Loop edges: player at x=-0.1, coach at x=14.2

SPINE = 7.75  # vertical spine for main flow

# ================================================================
# Title
# ================================================================
ax.text(7.0, 15.6, 'CRISP Pipeline', ha='center', fontsize=16, fontweight='bold', color=C_TEXT)
ax.text(7.0, 15.2, 'Coach generates  →  Players solve  →  Disagreement triggers discussion  →  All update via GRPO',
        ha='center', fontsize=9, color='#888888')

# ----------------------------------------------------------
# ROW 1: STEP 1 Coach + STEP 2 Players
# ----------------------------------------------------------

# Step 1: Coach (left column)
cw, ch = 3.2, 1.6
cx, cy = 0.8, 13.0
coach_right = cx + cw
coach_mid_y = cy + ch / 2
box(cx, cy, cw, ch, C_COACH, [
    ('STEP 1', 8, 'bold', '#8B6914'),
    ('Coach Generates', 9.5, 'bold', '#8B6914'),
    ('', 2, 'normal', C_TEXT),
    ('8 problems per batch', 8, 'normal', '#555'),
    ('Self-solves for ground truth', 8, 'normal', '#555'),
    ('Filters malformed outputs', 8, 'normal', '#555'),
])

# Step 2 label
ax.text(SPINE, 15.0, 'STEP 2: SOLVE', ha='center', fontsize=8, fontweight='bold', color='#1565C0')

# Alice
aw, ah = 4.5, 0.9
ax_a = 5.5
ay_a = 13.8
alice_mid_y = ay_a + ah / 2
box(ax_a, ay_a, aw, ah, C_PLAYER, [
    ('Alice    T=0.8, systematic', 8.5, 'bold', '#1565C0'),
    ('Qwen3-4B  ·  8 rollouts × 8 problems', 7.5, 'normal', '#555'),
])

# Bob
ax_b = 5.5
ay_b = 12.6
bob_mid_y = ay_b + ah / 2
box(ax_b, ay_b, aw, ah, C_PLAYER, [
    ('Bob    T=1.0, exploratory', 8.5, 'bold', '#1565C0'),
    ('Qwen3-4B  ·  8 rollouts × 8 problems', 7.5, 'normal', '#555'),
])

# Coach → Alice: horizontal right from coach, then vertical up, then horizontal to Alice
# Use 90° elbow: right to x=4.75, up to alice_mid_y, right to alice
elbow_x = 4.75
line(coach_right, cy + ch * 0.72, elbow_x, cy + ch * 0.72)  # horizontal from coach
line(elbow_x, cy + ch * 0.72, elbow_x, alice_mid_y)           # vertical up
arr(elbow_x, alice_mid_y, ax_a, alice_mid_y)                   # horizontal to alice

# Coach → Bob: horizontal right from coach, then vertical down, then horizontal to Bob
line(coach_right, cy + ch * 0.28, elbow_x, cy + ch * 0.28)   # horizontal from coach
line(elbow_x, cy + ch * 0.28, elbow_x, bob_mid_y)             # vertical down
arr(elbow_x, bob_mid_y, ax_b, bob_mid_y)                       # horizontal to bob

# ----------------------------------------------------------
# ROW 2: STEP 3 Verify (centered on spine)
# ----------------------------------------------------------
vw, vh = 5.5, 1.1
vx = SPINE - vw / 2
vy = 10.8
box(vx, vy, vw, vh, C_VERIFY, [
    ('STEP 3: VERIFY', 9, 'bold', C_TEXT),
    ('Extract \\boxed{} answers  ·  Check vs coach ground truth (SymPy)', 7.5, 'normal', '#555'),
])

# Alice → merge point → Verify
# Bob → merge point → Verify
# Both drop vertically to a horizontal merge line, then single arrow down to Verify
alice_bot = ay_a
bob_bot = ay_b
merge_y = 12.2  # horizontal merge line between Bob bottom and Verify top

alice_cx = ax_a + aw * 0.35  # alice drop point (left of center to avoid overlap)
bob_cx = ax_a + aw * 0.65   # bob drop point (right of center)

line(alice_cx, alice_bot, alice_cx, merge_y)  # alice drops
line(bob_cx, bob_bot, bob_cx, merge_y)        # bob drops
line(alice_cx, merge_y, bob_cx, merge_y)      # horizontal merge
arr(SPINE, merge_y, SPINE, vy + vh)           # single arrow to verify
# Connect merge line to spine
line(SPINE, merge_y, SPINE, merge_y)
# Actually need to connect the merge bar to spine properly
# The merge bar is from alice_cx to bob_cx, and SPINE is between them
# So the merge bar already covers SPINE. Just need the vertical arrow from merge_y down.

# ----------------------------------------------------------
# ROW 3: STEP 4 Disagree? (centered on spine)
# ----------------------------------------------------------
dw, dh = 4.0, 1.0
dx = SPINE - dw / 2
dy = 9.2
box(dx, dy, dw, dh, C_DECISION, [
    ('STEP 4: DISAGREE?', 9, 'bold', '#8B6914'),
    ('Majority vote per player', 7.5, 'normal', '#777'),
], border_color='#B8860B')

# Verify → Disagree
arr(SPINE, vy, SPINE, dy + dh)

# ----------------------------------------------------------
# Agree branch → right
# ----------------------------------------------------------
sw, sh = 2.3, 0.8
sx = 11.0
sy = dy + dh / 2 - sh / 2
box(sx, sy, sw, sh, C_SKIP, [
    ('Skip discussion', 8, 'normal', '#555'),
    ('r_solve only', 7, 'normal', '#888'),
])
arr(dx + dw, dy + dh / 2, sx, sy + sh / 2)
ax.text(dx + dw + 0.3, dy + dh / 2 + 0.25, 'Agree', ha='left', fontsize=7.5,
        color='#2E7D32', fontweight='bold')

# Disagree label
ax.text(SPINE + 0.3, dy - 0.25, 'Disagree', ha='left', fontsize=7.5,
        color='#C62828', fontweight='bold')

# ----------------------------------------------------------
# STEP 5: Discussion (centered on spine)
# ----------------------------------------------------------
d5w, d5h = 7.0, 1.4
d5x = SPINE - d5w / 2
d5y = 6.8
box(d5x, d5y, d5w, d5h, C_DISCUSS, [
    ('STEP 5: DISCUSSION', 9, 'bold', '#6A1B9A'),
    ('', 2, 'normal', C_TEXT),
    ('Select representative rollout per player', 8, 'normal', '#555'),
    ('Alice & Bob exchange reasoning  ·  Each evaluates peer\'s solution', 8, 'normal', '#555'),
])

arr(SPINE, dy, SPINE, d5y + d5h)

# ----------------------------------------------------------
# STEP 6: Re-answer (centered on spine)
# ----------------------------------------------------------
d6w, d6h = 7.0, 0.9
d6x = SPINE - d6w / 2
d6y = 5.2
box(d6x, d6y, d6w, d6h, '#E8E0F5', [
    ('STEP 6: RE-ANSWER', 9, 'bold', '#4A148C'),
    ('EVALUATION + FINAL ANSWER  ·  Verify against ground truth', 8, 'normal', '#555'),
])

arr(SPINE, d5y, SPINE, d6y + d6h)

# ----------------------------------------------------------
# STEP 7: Rewards
# ----------------------------------------------------------
ax.text(7.0, 4.65, 'STEP 7: COMPUTE REWARDS', ha='center', fontsize=10, fontweight='bold', color=C_TEXT)

# Player rewards (left)
prw, prh = 5.5, 1.3
prx = 0.5
pry = 2.8
pr_cx = prx + prw / 2  # 3.25
box(prx, pry, prw, prh, C_REWARD_P, [
    ('Player Rewards (per agent)', 9, 'bold', '#2E7D32'),
    ('', 2, 'normal', C_TEXT),
    ('r_correct + r_persuade − r_overlong', 8.5, 'normal', C_TEXT),
    ('Two-pool normalization · Independent EMA (η=0.2)', 7.5, 'normal', '#555'),
])

# Coach rewards (right)
crw, crh = 5.5, 1.3
crx = 8.0
cry = 2.8
cr_cx = crx + crw / 2  # 10.75
box(crx, cry, crw, crh, C_REWARD_C, [
    ('Coach Rewards (per problem)', 9, 'bold', '#E65100'),
    ('', 2, 'normal', C_TEXT),
    ('r_uncertainty + r_discussion − r_repetition', 8.5, 'normal', C_TEXT),
    ('max(0,·) floor · Sliding window buffer (W=10)', 7.5, 'normal', '#555'),
])

# Re-answer → Rewards: split into two clean 90° paths
split_y = d6y - 0.35  # horizontal rail below re-answer

# Left path: re-answer → player rewards
line(SPINE - 2.0, d6y, SPINE - 2.0, split_y)
line(SPINE - 2.0, split_y, pr_cx, split_y)
arr(pr_cx, split_y, pr_cx, pry + prh)

# Right path: re-answer → coach rewards
line(SPINE + 2.0, d6y, SPINE + 2.0, split_y)
line(SPINE + 2.0, split_y, cr_cx, split_y)
arr(cr_cx, split_y, cr_cx, cry + crh)

# Skip discussion → player rewards
# Goes from skip box bottom, straight down to a separate rail, then left to player rewards
skip_cx = sx + sw / 2  # 12.15
skip_rail_y = split_y + 0.4  # slightly above the re-answer rail to avoid overlap
line(skip_cx, sy, skip_cx, skip_rail_y, color='#999999', lw=1.2)
line(skip_cx, skip_rail_y, pr_cx, skip_rail_y, color='#999999', lw=1.2)
arr(pr_cx, skip_rail_y, pr_cx, pry + prh, color='#999999', lw=1.2)

# ----------------------------------------------------------
# GRPO Update boxes
# ----------------------------------------------------------
uw, uh = 5.0, 1.1

# Player update (left)
upx = 0.7
upy = 1.1
box(upx, upy, uw, uh, C_UPDATE_P, [
    ('Player GRPO Update', 10, 'bold', 'white'),
    ('Alice & Bob update weights independently', 8, 'normal', '#E8F5E9'),
], border_color='#2E7D32')

# Coach update (right)
ucx = 8.3
ucy = 1.1
box(ucx, ucy, uw, uh, C_UPDATE_C, [
    ('Coach GRPO Update', 10, 'bold', 'white'),
    ('Every iteration · Recalibrate difficulty', 8, 'normal', '#FFF3E0'),
], border_color='#E65100')

# Rewards → Updates (straight vertical)
arr(pr_cx, pry, pr_cx, upy + uh)
arr(cr_cx, cry, cr_cx, ucy + uh)

# ----------------------------------------------------------
# Loop arrows along far edges
# ----------------------------------------------------------

# Player loop: left edge
loop_p_x = -0.1
# From player update left edge, horizontal to loop edge, vertical up, horizontal into solve area
line(upx, upy + uh / 2, loop_p_x, upy + uh / 2, color=C_UPDATE_P, lw=2.5)
line(loop_p_x, upy + uh / 2, loop_p_x, 13.35, color=C_UPDATE_P, lw=2.5)
arr(loop_p_x, 13.35, ax_b, 13.35, color=C_UPDATE_P, lw=2.5)
ax.text(loop_p_x - 0.15, 7.5, 'Player loop — next iteration', ha='center', va='center',
        fontsize=7.5, color='#2E7D32', fontweight='bold', rotation=90)

# Coach loop: right edge
loop_c_x = 14.2
line(ucx + uw, ucy + uh / 2, loop_c_x, ucy + uh / 2, color=C_UPDATE_C, lw=2.5)
line(loop_c_x, ucy + uh / 2, loop_c_x, coach_mid_y, color=C_UPDATE_C, lw=2.5)
arr(loop_c_x, coach_mid_y, coach_right, coach_mid_y, color=C_UPDATE_C, lw=2.5)
ax.text(loop_c_x + 0.15, 7.5, 'Coach loop — recalibrate difficulty', ha='center', va='center',
        fontsize=7.5, color='#E65100', fontweight='bold', rotation=270)

plt.tight_layout(pad=0.3)
plt.savefig('/home/alex/mech_interp/CRISP/paper/pipeline.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved pipeline.png")
