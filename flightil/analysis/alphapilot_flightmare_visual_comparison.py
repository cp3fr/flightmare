import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def extract_gate_markers(frame):
    # find contours of blue gate markers
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
    filtered_contours = []
    for c in contours:
        if cv2.contourArea(c) > 1000.0:
            filtered_contours.append(c)
    # print(cv2.contourArea(contours[0]))
    contours = filtered_contours[::-1]
    # contours = contours[:2]

    # canny_output = cv2.Canny(frame_blue, 100, 100 * 2)

    # maybe compute convex hull?
    # fill in that area
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    # Draw contours + hull results
    drawing = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
    color = (255, 255, 255)
    for i in range(len(contours)):
        # cv2.drawContours(drawing, contours, i, color)
        # cv2.drawContours(drawing, hull_list, i, color)
        cv2.fillPoly(drawing, [hull_list[i]], color=color)

    """
    drawing_ff = drawing.copy()
    mask = np.zeros((drawing_ff.shape[0] + 2, drawing_ff.shape[1] + 2), dtype="uint8")
    test = list(zip(*np.where(drawing == 0)))
    test = test[int(len(test) / 2)]
    print(test)
    cv2.floodFill(drawing_ff, mask, (test[1], test[0]), 255)
    drawing_ff = cv2.bitwise_not(drawing_ff)

    drawing = cv2.bitwise_or(drawing, drawing_ff)
    """

    return drawing


# fov = 75
fov_all = []
nzob_all = []
app_all = {"mean": [], "std": [], "median": []}
fmp_all = {"mean": [], "std": [], "median": []}
ad_all = {"mean": [], "std": [], "median": []}
oup_all = {"mean": [], "std": [], "median": []}
colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
for fov in range(65, 86):
    fov_all.append(fov)

    print("FOV {}".format(fov))

    alphapilot_path = "/home/simon/Desktop/weekly_meeting/meeting14/alphapilot_original.mp4"
    flightmare_path = "/home/simon/Desktop/weekly_meeting/meeting14/flightmare_original_fov_{}.mp4".format(fov)

    show_frames = False

    ap_cap = cv2.VideoCapture(alphapilot_path)
    fm_cap = cv2.VideoCapture(flightmare_path)

    ap_params = tuple(ap_cap.get(i) for i in range(3, 8))
    fm_params = tuple(fm_cap.get(i) for i in range(3, 8))

    ap_range = 40  # 40 seems pretty good except for lighting changes (particularly for the middle gates)
    # ap_range = []
    ap_blue = np.array([136, 28, 0], dtype="uint8")
    # ap_low = ap_blue - [min(ap_range, ap_blue[i]) for i in range(3)]
    # ap_high = ap_blue + np.array([min(ap_range, 255 - ap_blue[i]) for i in range(3)], dtype="uint8")
    ap_low = np.array([90, 0, 0], dtype="uint8")
    ap_high = np.array([200, 80, 10], dtype="uint8")

    fm_range = 30  # 30 seems pretty good
    fm_blue = np.array([166, 66, 9], dtype="uint8")
    # fm_low = fm_blue - np.array([min(fm_range, fm_blue[i]) for i in range(3)], dtype="uint8")
    # fm_high = fm_blue + np.array([min(fm_range, 255 - fm_blue[i]) for i in range(3)], dtype="uint8")
    fm_low = np.array([130, 40, 0], dtype="uint8")
    fm_high = np.array([200, 100, 60], dtype="uint8")

    ret = True
    counter = 0
    ap_percentages = []
    fm_percentages = []
    ap_areas = []
    fm_areas = []
    ap_fm_differences = []
    overlap_union_percentages = []
    non_zero_overlaps_both = 0
    while ret:
        ap_ret, ap_frame = ap_cap.read()
        fm_ret, fm_frame = fm_cap.read()
        ret = ap_ret and fm_ret

        if ret:
            ap_frame = cv2.inRange(ap_frame, ap_low, ap_high)
            fm_frame = cv2.inRange(fm_frame, fm_low, fm_high)
            # print(ap_frame.max())
            frame = np.concatenate((ap_frame, fm_frame), axis=1)
            frame = extract_gate_markers(frame)

            # TODO: compare the overlaps
            ap_frame = frame[:, :800]
            fm_frame = frame[:, 800:]
            overlap = cv2.bitwise_and(ap_frame, fm_frame)
            union = cv2.bitwise_or(ap_frame, fm_frame)

            ap_filled = (ap_frame != 0).sum()
            fm_filled = (fm_frame != 0).sum()
            overlap_filled = (overlap != 0).sum()
            union_filled = (union != 0).sum()

            # print("AP: {:.2f}%, FM: {:.2f}%".format(ap_pct, fm_pct))
            if ap_filled != 0:
                ap_pct = 100.0 * overlap_filled / ap_filled
                ap_percentages.append(ap_pct)
                ap_areas.append(ap_filled)
            if fm_filled != 0:
                fm_pct = 100.0 * overlap_filled / fm_filled
                fm_percentages.append(fm_pct)
                fm_areas.append(fm_filled)
            if ap_filled != 0 and fm_filled != 0:
                ov_un_pct = 100.0 * overlap_filled / union_filled
                overlap_union_percentages.append(ov_un_pct)
                non_zero_overlaps_both += 1
                ap_fm_differences.append(ap_filled - fm_filled)

            if show_frames:
                cv2.imshow("frame", frame)
                cv2.imshow("both", overlap)
                """
                if counter == 100:
                    cv2.waitKey(0)
                """
                k = cv2.waitKey(50) & 0xff
                if k == 27:
                    break

            counter += 1

    if show_frames:
        cv2.destroyAllWindows()

    non_zero_pct = 100.0 * non_zero_overlaps_both / counter
    print("Non-zero overlaps for both images: {}/{} ({:.2f}%)\n".format(non_zero_overlaps_both, counter, non_zero_pct))

    # TODO: MAYBE ALSO NEED AREAS + AREA DIFFERENCES PER FRAME?

    fig, ax = plt.subplots(2, 3, figsize=(14, 6), dpi=100)

    # percentages
    ax[0][0].hist(ap_percentages, color=colors[0])
    ax[1][0].hist(fm_percentages, color=colors[0])
    ax[0][0].set_xlim(-5, 105)
    ax[1][0].set_xlim(-5, 105)
    ax[0][0].set_title("Percentage of overlap area to total")

    # areas
    ax[0][1].hist(ap_areas, color=colors[1])
    ax[1][1].hist(fm_areas, color=colors[1])
    ax[0][1].set_title("Gate marker area")

    # difference
    ax[0][2].hist(ap_fm_differences, color=colors[2])
    ax[0][2].set_title("Area differences")
    ax[1][2].axis("off")

    # other info
    nzob_all.append(non_zero_pct)

    app_all["mean"].append(np.mean(ap_percentages))
    app_all["std"].append(np.std(ap_percentages))
    app_all["median"].append(np.median(ap_percentages))

    fmp_all["mean"].append(np.mean(fm_percentages))
    fmp_all["std"].append(np.std(fm_percentages))
    fmp_all["median"].append(np.median(fm_percentages))

    ad_all["mean"].append(np.mean(ap_fm_differences))
    ad_all["std"].append(np.std(ap_fm_differences))
    ad_all["median"].append(np.median(ap_fm_differences))

    oup_all["mean"].append(np.mean(overlap_union_percentages))
    oup_all["std"].append(np.std(overlap_union_percentages))
    oup_all["median"].append(np.median(overlap_union_percentages))

    info_text = [
        "Non-zero overlap frames:\n{}/{} ({:.2f}%)".format(non_zero_overlaps_both, counter, non_zero_pct),
        "Overlap percentage AP:\nmean {:.2f}%, std {:.2f}%, median {:.2f}%".format(
            app_all["mean"][-1], app_all["std"][-1], app_all["median"][-1]),
        "Overlap percentage FM:\nmean {:.2f}%, std {:.2f}%, median {:.2f}%".format(
            fmp_all["mean"][-1], fmp_all["std"][-1], fmp_all["median"][-1]),
        "Area difference (AP - FM):\nmean {:.2f}, std {:.2f}, median {:.2f}".format(
            ad_all["mean"][-1], ad_all["std"][-1], ad_all["median"][-1]),
    ]

    for it_idx, it in enumerate(info_text):
        plt.gcf().text(0.7, 0.4 - 0.1 * it_idx, it, fontsize=12)
    # TODO: also give the means (medians?) and stds of all these measures

    fig.tight_layout()
    plt.savefig("/home/simon/Desktop/weekly_meeting/meeting14/analysis_fov_{:03d}.png".format(fov))
    # plt.show()
    plt.close()

ad_all["mean"] = np.array(ad_all["mean"])
ad_all["std"] = np.array(ad_all["std"])
ad_all["median"] = np.array(ad_all["median"])

app_all["mean"] = np.array(app_all["mean"])
app_all["std"] = np.array(app_all["std"])
app_all["median"] = np.array(app_all["median"])

fmp_all["mean"] = np.array(fmp_all["mean"])
fmp_all["std"] = np.array(fmp_all["std"])
fmp_all["median"] = np.array(fmp_all["median"])

oup_all["mean"] = np.array(oup_all["mean"])
oup_all["std"] = np.array(oup_all["std"])
oup_all["median"] = np.array(oup_all["median"])

# final plot with summaries over all FOVs (line plots showing mean, std, median)
fig, ax = plt.subplots(2, 2, figsize=(14, 8), dpi=100)

ax[0][0].plot(fov_all, nzob_all, color=colors[0], marker="o")
ax[0][0].plot(fov_all, oup_all["mean"], color=colors[1], marker="o")
ax[0][0].fill_between(fov_all, oup_all["mean"] - oup_all["std"],
                      oup_all["mean"] + oup_all["std"], color=colors[1], alpha=0.2)
ax[0][0].plot(fov_all, oup_all["median"], color=colors[1], linestyle="--", marker="v")
ax[0][0].set_ylim(-5, 105)
ax[0][0].set_xticks(fov_all)
ax[0][0].set_xticklabels([str(fov) for fov in fov_all])

ax[0][1].plot(fov_all, ad_all["mean"], color=colors[2], marker="o")
ax[0][1].fill_between(fov_all, ad_all["mean"] - ad_all["std"],
                      ad_all["mean"] + ad_all["std"], color=colors[2], alpha=0.2)
ax[0][1].plot(fov_all, ad_all["median"], color=colors[2], linestyle="--", marker="v")
ax[0][1].set_xticks(fov_all)
ax[0][1].set_xticklabels([str(fov) for fov in fov_all])

ax[1][0].plot(fov_all, app_all["mean"], color=colors[3], marker="o")
ax[1][0].fill_between(fov_all, app_all["mean"] - app_all["std"],
                      app_all["mean"] + app_all["std"], color=colors[3], alpha=0.2)
ax[1][0].plot(fov_all, app_all["median"], color=colors[3], linestyle="--", marker="v")
ax[1][0].set_ylim(-5, 105)
ax[1][0].set_xticks(fov_all)
ax[1][0].set_xticklabels([str(fov) for fov in fov_all])

ax[1][1].plot(fov_all, fmp_all["mean"], color=colors[4], marker="o")
ax[1][1].fill_between(fov_all, fmp_all["mean"] - fmp_all["std"],
                      fmp_all["mean"] + fmp_all["std"], color=colors[4], alpha=0.2)
ax[1][1].plot(fov_all, fmp_all["median"], color=colors[4], linestyle="--", marker="v")
ax[1][1].set_ylim(-5, 105)
ax[1][1].set_xticks(fov_all)
ax[1][1].set_xticklabels([str(fov) for fov in fov_all])

fig.tight_layout()
plt.show()
