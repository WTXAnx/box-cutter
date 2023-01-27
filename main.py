import cv2
from paddleocr import PaddleOCR
import os
import json

accecpt_prob = 0.95


locator_to_card_dx = -25
locator_to_card_dy = -25
card_lenx = 465
card_leny = 310

card_to_name_dx = 405
card_to_name_dy = 100
name_lenx = 30
name_leny = 170

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def calc_position_with_template(card_template, locator_template, name_template):
       global locator_to_card_dx, locator_to_card_dy, card_lenx, card_leny, card_to_name_dx, card_to_name_dy, name_lenx, name_leny
       card_lenx, card_leny = card_template.shape[:2]
       name_lenx, name_leny = name_template.shape[:2]
       print(card_lenx, card_leny, name_lenx, name_leny)

       locator_in_card = cv2.matchTemplate(card_template, locator_template, cv2.TM_CCOEFF_NORMED)
       _, max_val, _, max_loc = cv2.minMaxLoc(locator_in_card) #这个函数返回的坐标是反的很烦
       print("locator prob: ", max_val)
       locator_to_card_dy, locator_to_card_dx = max_loc #所以这里把坐标又反回去，下面同理
       locator_to_card_dx = -locator_to_card_dx
       locator_to_card_dy = -locator_to_card_dy
       print(locator_to_card_dx, locator_to_card_dy)
       
       name_in_card = cv2.matchTemplate(card_template, name_template, cv2.TM_CCOEFF_NORMED)
       _, max_val, _, max_loc = cv2.minMaxLoc(name_in_card)
       print("name prob: ", max_val)
       card_to_name_dy, card_to_name_dx = max_loc
       print(card_to_name_dx, card_to_name_dy)



def clean_set (s:set, glitch:int = 10):
       ls = sorted(list(s))
       if len(ls) == 0:
              return []
       pre = ls[0] - 100
       discard = set()
       for i in ls:
              if i <= pre + 10:
                     discard.add(i)
              else:
                     pre = i
       return s - discard

def exist_with_glitch(pos_set, new_pos, glitch = 10): #检查附近位置是否已经被识别到
       for old_pos in pos_set:
              ox, oy = old_pos
              nx, ny = new_pos
              if (abs(ox - nx) + abs(oy - ny) <= glitch):
                     return True
       return False

def image_to_position(origin_image, template):
       image_x, image_y = template.shape[:2]
       result = cv2.matchTemplate(origin_image, template, cv2.TM_CCOEFF_NORMED)
       x, y = result.shape[:2]
       max_prob = 0
       pos_set = set()
       result = result.tolist()
       for i in range(x):
              for j in range(y):
                     if result[i][j] > accecpt_prob:
                            max_prob = max(max_prob, result[i][j])
                            if not exist_with_glitch(pos_set, (i, j)):
                                   pos_set.add((i, j))

       print("max_prob:", max_prob)
       print("pos_set:", len(pos_set))
       print(pos_set)
       return pos_set

def cut_card(img_idx, origin_image, pos_set):
       for pos in pos_set:
              x, y = pos
              x += locator_to_card_dx
              y += locator_to_card_dy
              print(pos)
              print(x, y)
              if x < 0:
                     x = 0
              if y < 0:
                     y = 0
              if x + card_lenx > origin_image.shape[0] + 10 or y + card_leny > origin_image.shape[1] + 10:
                     continue
              if x + card_lenx > origin_image.shape[0]:
                     x = origin_image.shape[0] - card_lenx
              if y + card_leny > origin_image.shape[1]:
                     y = origin_image.shape[1] - card_leny
              timg = origin_image[x:x+card_lenx, y:y+card_leny]
              timg_name = timg[card_to_name_dx:card_to_name_dx+name_lenx, card_to_name_dy:card_to_name_dy+name_leny]

              text = ocr.ocr(timg_name, cls=True)
              if len(text) > 0 and len(text[0]) > 0:
                     name, _ = text[0][0][1]
                     cv2.imencode('.jpg', timg)[1].tofile(f'result\\{name}.jpg')
              else:
                     cv2.imwrite(f'result\\{img_idx}_{x}_{y}.jpg', timg)
       return

if __name__ == "__main__":
       with open("config.json", encoding='utf-8') as f:
              config = json.load(f)
       accecpt_prob = config.get('accecpt_prob')

       card_template = cv2.imread('template\\card.png')
       locator_template = cv2.imread('template\\locator.png')
       name_template = cv2.imread('template\\name.png')
       calc_position_with_template(card_template, locator_template, name_template)
       for root, dirs, files in os.walk('img'):
              for img_idx, f in enumerate(files):
                     img_path = os.path.join(root, f)
                     print(img_path)
                     origin_image = cv2.imread(img_path)
                     pos_set = image_to_position(origin_image, locator_template)
                     cut_card(img_idx, origin_image, pos_set)
       
       os.system('pause')






