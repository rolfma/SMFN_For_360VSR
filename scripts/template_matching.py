# opencv模板匹配----单目标匹配
import cv2
import os


def template_matching(target_name="", template_name="", other_dir=""):

    nonsuffixal_name = os.path.splitext(target_name)[0]
    suffixal_name = os.path.splitext(target_name)[1]
    target = cv2.imread(target_name)
    template = cv2.imread(template_name)
    theight, twidth = template.shape[:2]
    result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    target_cropped = target[min_loc[1]:min_loc[1] + twidth, min_loc[0]:min_loc[0] + theight]
    cv2.imwrite(nonsuffixal_name + "_cropped" + suffixal_name, target_cropped)

    cv2.rectangle(target, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 255, 0), 1)

    cv2.imwrite(nonsuffixal_name + "_marked" + suffixal_name, target)
    print("生成 " + nonsuffixal_name + "_cropped" + suffixal_name, end="\t")
    print(nonsuffixal_name + "_marked" + suffixal_name)

    if not other_dir == "":
        name_list = os.listdir(other_dir)
        for name in name_list:
            img_path = os.path.join(other_dir, name)
            other_img = cv2.imread(img_path)
            other_img = other_img[min_loc[1]:min_loc[1] + twidth, min_loc[0]:min_loc[0] + theight]
            nonsuffixal_name = os.path.splitext(img_path)[0]
            suffixal_name = os.path.splitext(img_path)[1]
            cv2.imwrite(nonsuffixal_name + "_cropped" + suffixal_name, other_img)

            print("生成 " + nonsuffixal_name + "_cropped" + suffixal_name)


if __name__ == "__main__":
    template_matching(target_name="misc/target/034.png",
                      template_name="misc/template/034temp.png",
                      other_dir="misc/other_dir/9")
