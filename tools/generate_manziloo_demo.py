from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import subprocess
from arabic_reshaper import reshape
from bidi.algorithm import get_display

W,H=1280,720
OUT=Path('manziloo-chatgpt-demo.mp4')
D=Path('demo_frames'); D.mkdir(exist_ok=True)
REG='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
BG='#F7F7F5'; WHITE='#FFFFFF'; INK='#161616'; MUTED='#6C6C6C'; GREEN='#164E3D'; GREEN2='#E8F2ED'; BORDER='#D9D9D6'; BLUE='#EAF2FF'; RED='#9B2C2C'

def f(s,b=False): return ImageFont.truetype(BOLD if b else REG,s)
def ar(s): return get_display(reshape(s))
def rr(d,xy,r=24,fill=WHITE,outline=None,width=1): d.rounded_rectangle(xy,radius=r,fill=fill,outline=outline,width=width)
def txt(d,xy,s,fs,b=False,fill=INK,anchor='la'): d.text(xy,s,font=f(fs,b),fill=fill,anchor=anchor)
def artxt(d,xy,s,fs,b=False,fill=INK,anchor='ra'): d.text(xy,ar(s),font=f(fs,b),fill=fill,anchor=anchor)
def base(step):
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    d.rectangle([0,0,W,76],fill=WHITE); d.line([0,76,W,76],fill=BORDER,width=2)
    rr(d,[32,18,72,58],12,GREEN); d.polygon([(42,42),(52,31),(62,42),(62,51),(55,51),(55,43),(49,43),(49,51),(42,51)],fill=WHITE)
    txt(d,(90,38),'Manziloo | ',26,True,anchor='lm'); artxt(d,(315,38),'منزلو',26,True,anchor='rm')
    txt(d,(W-32,38),'OpenAI review demo',18,fill=MUTED,anchor='rm'); txt(d,(W-32,64),step,13,fill=MUTED,anchor='rd')
    txt(d,(W//2,H-18),'Simulation of the intended ChatGPT Developer Mode workflow',13,fill=MUTED,anchor='mm')
    return im,d

def save(i,im): p=D/f'scene_{i:02d}.png'; im.save(p); return p
sc=[]

im,d=base('1 / 8'); rr(d,[130,155,1150,565],32,WHITE,BORDER,2); txt(d,(640,225),'Manziloo | منزلو',54,True,GREEN,'mm'); txt(d,(640,295),'ChatGPT MCP service-request demo',30,True,anchor='mm'); txt(d,(640,365),'Simulation only — synthetic demo data',27,True,RED,'mm'); txt(d,(640,420),'No live customer data is used in this recording.',22,fill=MUTED,anchor='mm'); txt(d,(640,475),'MCP: https://manziloo.com/mcp',22,anchor='mm'); sc.append((save(1,im),6))

im,d=base('2 / 8'); txt(d,(90,126),'Chat conversation',24,True); rr(d,[90,180,1190,590],26,WHITE,BORDER,2); rr(d,[450,235,1125,350],24,GREEN2); artxt(d,(1090,282),'أبي فني مكيف بالرياض، المكيف شغال لكن ما يبرد',30,True,anchor='ra'); txt(d,(1088,326),'User',15,fill=MUTED,anchor='ra'); rr(d,[150,400,905,520],24,'#F1F1EF'); txt(d,(185,430),'Manziloo',18,True,GREEN); artxt(d,(865,473),'أفتح لك نموذج طلب خدمة في الرياض. الإرسال لا يعني تأكيد فني أو موعد.',25,anchor='ra'); sc.append((save(2,im),7))

im,d=base('3 / 8'); txt(d,(90,120),'Tool invoked: request_service',24,True,GREEN); rr(d,[90,160,1190,620],26,WHITE,BORDER,2); artxt(d,(1120,205),'طلب خدمة منزلو',34,True,GREEN,'ra'); txt(d,(120,218),'Riyadh-only intake form',17,fill=MUTED)
for y,label,val in [(265,'الاسم','عميل تجريبي'),(353,'رقم الجوال','0500000000'),(441,'الحي','الياسمين')]: rr(d,[690,y,1110,y+72],14,BG,BORDER); artxt(d,(1085,y+18),label,14,fill=MUTED,anchor='ra'); artxt(d,(1085,y+52),val,21,True,anchor='ra') if label!='رقم الجوال' else txt(d,(1085,y+52),val,21,True,anchor='ra')
rr(d,[120,265,640,337],14,BG,BORDER); artxt(d,(615,283),'نوع العقار',14,fill=MUTED,anchor='ra'); artxt(d,(615,317),'شقة',21,True,anchor='ra'); rr(d,[120,353,640,425],14,BG,BORDER); artxt(d,(615,371),'الفئة',14,fill=MUTED,anchor='ra'); artxt(d,(615,405),'تبريد وتكييف',21,True,anchor='ra'); rr(d,[120,441,640,513],14,BG,BORDER); artxt(d,(615,459),'تفاصيل العمل',14,fill=MUTED,anchor='ra'); artxt(d,(615,495),'المكيف شغال لكن ما يبرد',20,True,anchor='ra'); rr(d,[120,535,1110,588],14,BLUE,'#B8CCE8'); txt(d,(145,562),'Read-only step: no request has been submitted yet.',21,True,'#244F7A','lm'); sc.append((save(3,im),8))

im,d=base('4 / 8'); txt(d,(90,120),'Explicit confirmation before write action',24,True,GREEN); rr(d,[120,175,1160,585],26,WHITE,BORDER,2); artxt(d,(1110,225),'مراجعة الطلب قبل الإرسال',32,True,anchor='ra');
for y,s in [(300,'الفئة: تبريد وتكييف'),(345,'المدينة: الرياض'),(390,'الحي: الياسمين'),(435,'الوصف: المكيف شغال لكن ما يبرد')]: artxt(d,(1080,y),s,24,anchor='ra')
d.rectangle([180,470,210,500],outline=GREEN,width=3,fill=WHITE); d.line([187,485,196,494],fill=GREEN,width=4); d.line([196,494,207,477],fill=GREEN,width=4); artxt(d,(1060,486),'أوافق على سياسة الخصوصية واستخدام بيانات التواصل لمتابعة الطلب',21,anchor='ra'); rr(d,[420,525,860,575],18,GREEN); artxt(d,(820,550),'تأكيد وإرسال الطلب',22,True,WHITE,'ra'); sc.append((save(4,im),7))

im,d=base('5 / 8'); txt(d,(90,120),'Tool invoked: submit_service_request',24,True,GREEN); rr(d,[115,175,1165,585],26,WHITE,BORDER,2); rr(d,[170,220,1110,300],18,GREEN2); txt(d,(210,260),'✓ request_received',31,True,GREEN,'lm'); txt(d,(210,350),'Demo reference',17,fill=MUTED); txt(d,(210,390),'DEMO-REQUEST-001',30,True); txt(d,(210,455),'Recorded in Manziloo Ops for follow-up.',23); txt(d,(210,500),'Idempotent submission: retrying the same request does not create a duplicate.',20,fill=MUTED); sc.append((save(5,im),7))

im,d=base('6 / 8'); txt(d,(90,120),'Important customer-facing boundary',24,True,GREEN); rr(d,[120,175,1160,585],26,WHITE,BORDER,2); artxt(d,(1100,230),'تم استلام طلبك للمتابعة من منزلو',34,True,GREEN,'ra'); txt(d,(180,300),'The request is received. The following are NOT confirmed:',24,True)
for y,s in [(360,'Technician'),(415,'Appointment time'),(470,'Fee / waiver decision'),(525,'Payment')]: txt(d,(210,y),'×',34,True,RED,'lm'); txt(d,(250,y),s,26,anchor='lm')
artxt(d,(1080,560),'سيتواصل فريق منزلو لتحديد الخطوة التالية.',22,anchor='ra'); sc.append((save(6,im),7))

im,d=base('7 / 8'); txt(d,(90,120),'MCP integration summary',24,True,GREEN); rr(d,[100,165,1180,595],26,WHITE,BORDER,2); txt(d,(145,220),'Endpoint',17,fill=MUTED); txt(d,(145,255),'https://manziloo.com/mcp',27,True); txt(d,(145,330),'Tools',17,fill=MUTED); rr(d,[145,360,600,425],14,GREEN2); txt(d,(175,392),'request_service',24,True,GREEN,'lm'); txt(d,(630,392),'read-only form open/prefill',20,fill=MUTED,anchor='lm'); rr(d,[145,455,600,520],14,GREEN2); txt(d,(175,487),'submit_service_request',24,True,GREEN,'lm'); txt(d,(630,487),'private request_received write',20,fill=MUTED,anchor='lm'); sc.append((save(7,im),7))

im,d=base('8 / 8'); rr(d,[150,165,1130,565],30,WHITE,BORDER,2); txt(d,(640,235),'Demo complete',45,True,GREEN,'mm'); artxt(d,(640,320),'منزلو — خدمات منزلية في الرياض',34,True,anchor='mm'); txt(d,(640,390),'request_service → explicit confirmation → submit_service_request',23,anchor='mm'); txt(d,(640,455),'Result: request_received',27,True,anchor='mm'); txt(d,(640,510),'No appointment or payment is confirmed by the plugin.',20,fill=MUTED,anchor='mm'); sc.append((save(8,im),6))

concat=D/'concat.txt'
with concat.open('w') as h:
    for p,dur in sc: h.write(f"file '{p.resolve()}'\nduration {dur}\n")
    h.write(f"file '{sc[-1][0].resolve()}'\n")
subprocess.run(['ffmpeg','-y','-f','concat','-safe','0','-i',str(concat),'-vf','fps=30,format=yuv420p','-c:v','libx264','-preset','veryfast','-crf','28','-movflags','+faststart',str(OUT)],check=True)
print(OUT)
