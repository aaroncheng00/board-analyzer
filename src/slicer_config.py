from dataclasses import dataclass

@dataclass
class SlicerConfig: 
    # --- Step 1: find_board_bbox ---
    bbox_canny_low: int = 50
    bbox_canny_high: int = 150
    bbox_min_area_frac: float = 0.05   # reject contours smaller than this
                                        # fraction of the full image area
    bbox_dilate_kernel: int = 3        # square kernel used to connect broken /
                                        # anti-aliased border edges into one
                                        # continuous contour
    bbox_dilate_iterations: int = 1
    bbox_aspect_tol: float = 0.95      # accept a contour as board-like when its
                                        # w/h ratio falls within [tol, 1/tol].
                                        # Boards are usually close to square

    # --- Step 2: detect_grid_lines (Hough-based line detection) ---
    line_canny_low: int = 30
    line_canny_high: int = 100
    min_line_frac: float = 0.35        # min line length, as a fraction of
                                        # board width/height. Kept low since
                                        # thin/anti-aliased gridlines often
                                        # produce a broken, shorter-than-true
                                        # edge run (see detect_grid_lines
                                        # docstring for the full explanation)
    max_line_gap_frac: float = 0.15    # max gap HoughLinesP bridges within
                                        # one line, as a fraction of the
                                        # board's smaller dimension
    angle_tol_deg: float = 3           # how close to perfectly horizontal/
                                        # vertical a line must be
    line_run_frac: float = 0.8         # a detected segment must span at least
                                        # this fraction of min_line_length
                                        # along its own axis to be counted
    hough_threshold_frac: float = 0.05 # hough_threshold auto-derived as
                                        # max(hough_threshold_min,
                                        #     min_line_length * this).
                                        # Was 0.4, which detected 13 rows on an
                                        # 8x8 chessboard: piece artwork produces
                                        # extra horizontal edge runs, and a high
                                        # threshold changes which segments Hough
                                        # extracts (it consumes edge pixels as
                                        # it goes, so the response is not
                                        # monotonic). 0.02-0.15 is a broad
                                        # plateau where both test boards come
                                        # out correct; 0.05 sits inside it
    hough_threshold_min: int = 20      # in practice THIS is the binding value:
                                        # at frac=0.05 the derived term is ~13
                                        # (chess) and ~7 (tango), so the floor
                                        # wins on both. Tune this rather than
                                        # the fraction above
    cluster_gap_frac: float = 0.02     # nearby detected line positions
                                        # within this fraction of board size
                                        # get merged into one
    cluster_gap_min: int = 4           # floor for the above, in pixels
    # --- grid detection strategy ---
    grid_detector: str = "profile"     # "profile" or "hough".
                                        # hough looks for DRAWN gridlines, which
                                        # chess_2 does not have -- its squares
                                        # differ only in wood shade. No hough
                                        # parameter setting handles both chess
                                        # boards: chess_1 needs a large
                                        # max_line_gap_frac to bridge its
                                        # fragmented lines, chess_2 needs a small
                                        # one or grain gets stitched into lines.
                                        # "profile" keys on periodicity instead
                                        # and handles all three boards. Kept
                                        # switchable so the two stay comparable
    profile_n_min: int = 3             # cell-count search range
    profile_n_max: int = 20
    profile_peak_tol_frac: float = 0.04 # window when sampling a predicted
                                        # boundary, as a fraction of CELL size,
                                        # absorbing rounding and inset borders.
                                        # Relative rather than fixed px because
                                        # cells range 70-180px here: a flat 2px
                                        # is 2.9% of a tango cell but 1.1% of a
                                        # chess_2 one. It was 2px, and queens_1
                                        # failed outright -- its bbox includes
                                        # the grid's outer border, so the
                                        # interior is inset and an even division
                                        # drifts ~2.8px at the first and last
                                        # boundary. Those two read exactly 0.0,
                                        # the percentile picked one up, and
                                        # every N scored ~0.003
    profile_peak_tol_min: int = 2      # floor in px, for very small cells
    profile_percentile: float = 20.0   # how the per-boundary strengths are
                                        # aggregated into one score for a
                                        # candidate N. 100 = max, 50 = median,
                                        # 0 = min.
                                        # mean and min both fail: mean scores a
                                        # single bright line at 0.33 (it averages
                                        # one real hit with empty predictions, so
                                        # a non-grid passes), while min picks 3
                                        # instead of 6 on tango_1 columns, where
                                        # the weakest genuine boundary sits at
                                        # 86% of the mean. The 20th percentile
                                        # tolerates a weak boundary or two and
                                        # still requires most predictions to
                                        # land on something real
    profile_max_inset_frac: float = 0.05 # the grid may sit this far inside the
                                        # crop at either end. find_board_bbox
                                        # finds the board INCLUDING any frame:
                                        # checkers_1's wooden frame leaves the
                                        # real grid at 24..1159 of a 1184px
                                        # crop, so assuming it spans the full
                                        # crop gives 148px cells where the truth
                                        # is 142. This bound also stops a small
                                        # N cheating -- with the offset fully
                                        # free, N=3 finds a lucky placement on
                                        # every board. 0.02-0.08 all work; 0.12
                                        # breaks queens_1 columns
    profile_inset_step: int = 2        # px granularity of the coarse sweep
    profile_refine_px: int = 8         # +/- window for the asymmetric refine
                                        # after the symmetric coarse pass
    profile_norm_percentile: float = 99.0 # ceiling used to normalise the
                                        # profile. NOT the max: one strong edge
                                        # -- a board frame, a UI highlight --
                                        # otherwise squashes every real
                                        # gridline. Both checkers boards scored
                                        # 0.19 against a 0.20 threshold for
                                        # exactly this reason; at p99 they score
                                        # 1.00
    profile_min_score: float = 0.50    # best-N score below this means no grid.
                                        # Needed because when every score is ~0
                                        # the largest-N rule returns
                                        # profile_n_max, which is how a single
                                        # bright line was passing as a grid.
                                        # With p99 normalisation real boards
                                        # score 0.96-1.00 and adversarial images
                                        # that clear the contrast guard score
                                        # <=0.20, so 0.50 sits in a wide gap --
                                        # it was 0.20, one hundredth away from
                                        # rejecting both checkers boards
    profile_rel_threshold: float = 0.90 # take the LARGEST N scoring within this
                                        # fraction of the best. Divisors of the
                                        # true N tie exactly (their boundaries
                                        # are a genuine subset); multiples score
                                        # lower by adding weak midpoints
    profile_min_contrast: float = 5.0  # peak/median of the RAW profile. Below
                                        # this the image has no grid signal and
                                        # detection raises instead of returning
                                        # profile_n_max, which is what it would
                                        # otherwise do on a blank image.
                                        # Real boards measure 36-547; uniform
                                        # images measure ~1

    uniform_spacing: bool = True       # respace the detected lines evenly once
                                        # their COUNT is known, instead of
                                        # trusting individual positions. A
                                        # grid-size override that skips
                                        # inference entirely is the fully
                                        # robust fix and is not built
