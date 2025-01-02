"""
2. 주요 강의 내용
- 거래전략별로 설정된 화면에 대한 이해 및 이와 연관된 코드를 확인해 본다.
- 이베스트에서 제공하는 검색식을 API를 통하여 살펴본다.
- 화면에 있는 거래시간, 일괄 매도 시간 처리 방법에 대해서 알아본다.
- OnClockTick 의 역할에 대해서 알아본다.
- 주문 낼 때의 유의 사항에 대해서 알아보고 해당 종목에 lock을 거는 방벙에 대해서 알아본다. (미체결 상태, 주문 대기 상태 등)
- 종목검색 API에 대해서 알아보고 해당 종목을 로봇에 걸어 시세를 받는다.
- 종목검색식 결과를 처음 수신할 때와 이후 실시간 수신시 차이를 알아본다. (OnReceiveData, OnReceiveSearchData)
- uuid 식별자를 통하여 로봇별로 분개한다.
- API를 이용하여 주문시 의외로 오류가 날 상황이 많이 있다. 사용자들의 제보를 통한 해당 상황을 살펴보고 오주문 방지 코드를 확인한다.
- mymoneybot과 로봇간의 함수 또는 데이터 공유 방식에 대해서 알아본다.
- 데이터를 DB에 읽고 쓸 때 유의 사항에 대해 알아본다.
- 포트폴리오 및 포트폴리오 제어, 포트폴리오별 주문 제어에 대해서 알아보고 포트폴리오 주문시 나타날 수 있는 오류를 예방한다.
- API에 요청할 데이터를 등록하고 수신한다. 수신 해지 방법 및 이 경우 발생할 수 있는 오류에 대해서 알아보고 방어하는 코드를 삽입한다.


 2022.11.27
 - 기준봉 조건을 조건검색의 의한 기준봉으로 매매를 한다.
 - 기준봉 기준 10일 이전 가격이 상승 추세만 판단 한다.


 2023.01.06
  - 추적법 차트 인덱스로 최대 500일까지만적용이 되어..
  - 종료일자를 희망날짜로 입력하면 그 뒤로 500 뒤 까지 데이터 불러온다.
  
  2023.03.02
   - VI 종목 매매 시 실시간 추적법 지지선 2번 매매 기능 추가
   - 차트인덱스 500일 x 3회 반복에서  2회로 변경

   2023.03.11
    - vi 종목 중에 음봉 2%? 이하 확인 후 아래 추적법 지지선 매매
    - 1차 매매 후 2차 지지선까지 매매하도록 함.
    - VI 발동 후 10분이내에서 매매 안되면 매수 취소 시킴.

    2023.03.14 / ver.2.0 시작
     -vi 발동   1분봉 -2% 확인 기능 추가
     - 600초 대기 후 취소 주문 기능 추가

    2023.03.15
     - 1,2차 매수 기능,
     - 연속 음봉 발생 시 합하여 -2% 이하인지 판단 기능
     - self.오늘가격 초기값 = 1로 세팅

    2023.03.16
     - 딕셔너리 검토 필요.
    - 실시간추적법 디버깅 파일 참고
    - 계좌에 있던 종목코드로 딕셔너리 오류 발생 - 딕셔너리 코드와 현재 종목코드가 계속
    - 1분봉, 만들기,
    - 딕셔너리 코드 에러 확인


    2023.03.19
    - 1분동 만들기 수정- 등락률 분석?

    2023.03.25
     - 1분봉 t1302 사용
     최대 10번까지 매분 요청하여 등락률 확인

    2023.03.26
     - 1분봉 쿼리 기능 삭제..
     그냥 리얼데이터로 검토 할것.
     : 30초봉으로 설정.
     - 매수 2번 하는 기능 추가 할것

    2023.03.29 /
    - 기존 2번 매수로 원복
    - todo 지지가격 0.5% 이내 통합 기능 진행 중....

    2023.03.27
     - 2차 매수안한것 1차 매도 시 취소 시킴.
     - 추적법 지지선 보정 할것 !
      : 지지가격이 0.5 % 이내면 평균값으로 통합 시킴.

    2023.03.28
     - VI 발동 후 상위 지지선 돌파 후 매수 기능 추가 ???


    2023.03.31 상승하향 통합버전

    - 매수 조건이 VI 가격이 아니고 VI 가격 상위 첫 지지선 가격으로 변경
    #상향 매수 고전
        # VI 가격 위 첫 저항 돌파 후 2% 상승 시 지지선 매수 방법
    # 하향 매수 조건
        # VI 발동 후 하락 2% 이하면 그 30초봉 종가 가격 아래 첫 지지선 가격 매수


    2023.04.11
     - 상위 매매는 +100 % 로 설정
     - 하향은 -3%로 조건으로 변경
     - vi 발동 후 30분까지 감시.

    2023.05.17
     # self.차트인덱스종목리스트[self.현재종목코드][i+1] = (self.차트인덱스종목리스트[self.현재종목코드][i + 1] + self.차트인덱스종목리스트[self.현재종목코드][i]) / 2
       --> 변경     : self.차트인덱스종목리스트[self.현재종목코드][i+1] = self.차트인덱스종목리스트[self.현재종목코드][i]


    #todo 주기적으로 아래 실행 할것 리스트 항목 제거 목적....
    #todo: 테스트용 전체 삭제
      self.주문기록삭제('')
      **************************************************
"""


import win32com.client
import pythoncom

import os, sys
sys.path.append('../../..')
import uuid

import pandas as pd
import pandas.io.sql as pdsql
# from pandas import DataFrame, Series
from pandas import DataFrame, Series, Panel
# from pandas.lib import Timestamp
import numpy as np

import sqlite3
import time
# from datetime import datetime, timedelta
# import datetime
import math

import PyQt5
from PyQt5 import QtCore, QtGui, uic
from PyQt5 import QAxContainer
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow, QDialog, QMessageBox, QProgressBar)
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *

from XASessions import *
from XAQuaries import *
from XAReals import *

from CRobot import *
from myStockClass import *
from Utils import *

# import pandas as pd
# from pandas import DataFrame, Series, Panel
# import numpy as np
# import pandas as pd
# import time
import datetime
# import ccxt
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.signal import argrelextrema, find_peaks
from sklearn.neighbors import KernelDensity

# import numpy as np
# import pandas as pd
# import time
# import datetime
# import ccxt
# import matplotlib.pyplot as plt
# import hdbscan
# from sklearn.preprocessing import StandardScaler, RobustScaler

__PATHNAME__ = os.path.dirname(sys.argv[0])
__PLUGINDIR__ = os.path.abspath(__PATHNAME__)

ROBOT_NAME = "실시간추적법"

Ui_실시간추적법, QtBaseClass_실시간추적법 = uic.loadUiType("%s\\Plugins\\실시간추적법.ui" % __PLUGINDIR__)
class CUI실시간추적법(QDialog, Ui_실시간추적법):
    def __init__(self, parent=None):
        super(__class__, self).__init__(parent)
        # self.setAttribute(Qt.WA_DeleteOnClose)
        self.setupUi(self)
        self.parent = parent

    def SearchFile(self):
        pathname = os.path.dirname(sys.argv[0])
        RESDIR = "%s\\ACF\\" % os.path.abspath(pathname)

        fname = QFileDialog.getOpenFileName(self, 'Open file',RESDIR, "조검검색(*.acf)")
        self.lineEdit_filename.setText(fname[0])


class 실시간추적법(CRobot, myStockClass):
    def instance(self):
        UUID = uuid.uuid4().hex
        return 실시간추적법(Name=ROBOT_NAME, UUID=UUID)

    def __init__(self, Name, UUID):
        super(__class__, self).__init__(Name, UUID)
        self.parent = None

        self.단위투자금 = 100 * 10000    # 100 만원
        # self.매수방법 = '00'            # 지정가
        self.매수방법 = '03'            # 시정가
        self.매도방법 = '03'            # 시장가
        self.포트폴리오수 = 2
        self.profitstop = 0.01
        self.losscut = 0.03
        self.targetPercent = -0.02      # -1% vi 발동 후 -x% 떨어진 후 지지값 산출
        self.targetSeconds = 600*12       # 10분동안 위 1분봉 -2% 이하가 발생하지 않으면 매수취소 시킴.
        self.lossbuy = 0.02
        self.ACF파일 = ''
        self.일괄매도시각 = '23:00:00'
        # todo 기준봉 매매법에는 10시 30분까지만 매수 가능 이후 시간에는 미체결은 매수취소 시킴.
        self.일괄취소시각 = '15:05:00'
        self.매수거래시간STR = '''09:10:10-23:00:00'''
        self.매도거래시간STR = '''09:00:00-15:00:00'''
        self.매수거래중 = False
        self.매도거래중 = False
        self.금일매도종목 = []
        self.계좌조회종목 = dict()

        self.주문번호리스트 = []
        self.매수Lock = dict()
        self.매도Lock = dict()
        self.종목수인덱스 = None

        self.QueryInit()

        #내꺼
        # myStockClass = myStockClass()   #객체생성
        # self.slice_ = slice(0, 1500)
        # self.peaks_range = [5, 10]
        # self.num_peaks = -999

        # self.price_range = None
        # self.peaks = []  # peaks 변수를 초기화
        # interval = extrema_prices[0] / 10000
        # self.support_resistance(code, nowPrice, prePrice)

        self.buy_prices = dict()
        self.crossPrice = dict()
        self.이전가격 = dict()
        self.현재가격 = dict()
        self.realNowCode = None
        self.기준봉검색종목 = []
        self.기준봉선정종목 = []
        self.기준봉종목수 = None
        self.현재종목코드 = None
        self.현재차트인덱스종목코드 = ''
        self.차트인덱스종목리스트 = dict()
        self.가우시안가격리스트 = dict()
        self.VIsticList = dict()
        self.주문번호관리리스트 = dict()
        self.myChartCnt = dict()
        self.myN분봉Cnt = dict()
        self.미체결리스트 = dict()
        self.오늘가격 = dict()
        self.총매수량 = dict()
        self.highPercent = 2     # 2% 상위값
        self.lowPercent = -2      # -2% 하위값.
        self.StartStock = True
        self.checkRobotCnt = 1000
        self.gDf = None
        self.테스트용 = 0
        self.Qwaitting = False
        self.minuteTime = 0
        self.siga = 0
        self.jongga = 0
        self.staticrate = 0

        self.clock = None
        self.전량매도 = False
        self.전량취소 = True
        self.미체결용 = False

    def QueryInit(self):
        self.XQ_t1857 = None
        self.XR_S3_ = None
        self.XR_K3_ = None
        self.QA_CSPAT00600 = None
        self.QA_CSPAT00800 = None
        self.XQ_t0425 = None
        self.XQ_t1302 = None
        self.XQ_t8412 = None
        # self.XR_SC0 = None # 접수
        self.XR_SC1 = None # 체결
        # self.XR_SC2 = None # 정정
        # self.XR_SC3 = None # 취소
        # self.XR_SC4 = None # 거부

    def modal(self, parent):
        ui = CUI실시간추적법(parent=parent)
        ui.setModal(True)

        ui.lineEdit_name.setText(self.Name)
        ui.lineEdit_unit.setText(str(self.단위투자금 // 10000))
        ui.lineEdit_profitstop.setText(str(self.profitstop))
        ui.lineEdit_losscut.setText(str(self.losscut))
        ui.lineEdit_portsize.setText(str(self.포트폴리오수))
        ui.comboBox_buy_sHogaGb.setCurrentIndex(ui.comboBox_buy_sHogaGb.findText(self.매수방법, flags=Qt.MatchContains))
        ui.comboBox_sell_sHogaGb.setCurrentIndex(ui.comboBox_sell_sHogaGb.findText(self.매도방법, flags=Qt.MatchContains))
        ui.lineEdit_filename.setText(self.ACF파일)
        ui.plainTextEdit_buytime.setPlainText(self.매수거래시간STR)
        ui.plainTextEdit_selltime.setPlainText(self.매도거래시간STR)
        ui.lineEdit_sellall.setText(self.일괄매도시각)

        r = ui.exec_()
        if r == 1:
            self.Name = ui.lineEdit_name.text().strip()
            self.단위투자금 = int(ui.lineEdit_unit.text().strip()) * 10000
            self.매수방법 = ui.comboBox_buy_sHogaGb.currentText().strip()[0:2]
            self.매도방법 = ui.comboBox_sell_sHogaGb.currentText().strip()[0:2]
            self.포트폴리오수 = int(ui.lineEdit_portsize.text().strip())
            self.ACF파일 = ui.lineEdit_filename.text().strip()
            self.profitstop = float(ui.lineEdit_profitstop.text().strip())
            self.losscut = float(ui.lineEdit_losscut.text().strip())
            self.매수거래시간STR = ui.plainTextEdit_buytime.toPlainText().strip()
            self.매도거래시간STR = ui.plainTextEdit_selltime.toPlainText().strip()

            매수거래시간1 = self.매수거래시간STR
            매수거래시간2 = [x.strip() for x in 매수거래시간1.split(',')]

            result = []
            for temp in 매수거래시간2:
                result.append([x.strip() for x in temp.split('-')])

            self.매수거래시간 = result

            매도거래시간1 = self.매도거래시간STR
            매도거래시간2 = [x.strip() for x in 매도거래시간1.split(',')]

            result = []
            for temp in 매도거래시간2:
                result.append([x.strip() for x in temp.split('-')])

            self.매도거래시간 = result

            self.일괄매도시각 = ui.lineEdit_sellall.text().strip()

        return r


    def support_resistance(self, code,nowPrice, prePrice):
        # 특정 가격 리스트
        # target_prices = [100, 110, 120, 130, 140]

        if self.가우시안가격리스트 is None:
            print("가우시안가격리스트 is not initialized")
            return
        else:
            try:
                # print("55 ", self.가우시안가격리스트[code])
                # target_prices = self.가우시안가격리스트[code]
                # 가우시안가격리스트에서 코드 조회 시
                target_prices = self.가우시안가격리스트.get(code, [])
            except Exception as e:
                print(f"55Error in support_resistance: {e}")

        try:
            previous_price = prePrice  # 이전 가격 (예시로 설정)
            current_price = nowPrice  # 현재 가격 (예시로 설정)
            # print("55 ", previous_price, current_price)
        except Exception as e:
            print(f"77Error in support_resistance: {e}")

        ydm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mm = datetime.datetime.now()
        dd = "{:%Y-%m-%d}".format(datetime.datetime.now())

        # 현매수량 = ((self.단위투자금) // nowPrice)
        현매수량 = 1  # 테스트용

        # crossPrice가 없는 경우 기본값 설정
        if code not in self.crossPrice:
            self.crossPrice[code] = 0

        # print(f"2번째 지지저항_ 현재종목코드 = {code}, 현재가격 = {current_price}, 이전가격 = {previous_price} 크로스자격={self.crossPrice[code]}")
        # return 0
        # 해당 종록의 돌파가격 초기화. 첫 돌파가격은 vi 이후 첫번째 가격이 불러왔을 때 이 가격 첫번ㅉ 아래 가격 지정.
        # 이유는 처음에 상승이 아닌 계속 하락할 경우를 대비해서...
        # self.crossPrice[code] 가 등록이 안되었을때 대응 필요
        if self.crossPrice[code] == 0:
            firstPrice = sorted([price for price in target_prices if price < current_price], reverse=True)
            # self.crossPrice[code].append(firstPrice)
            self.crossPrice[code] = firstPrice[0]
            print(
                f"코로스 내부 현재종목코드 = {code}, 현재가격 = {current_price}, 이전가격 = {previous_price} 크로스자격={self.crossPrice[code]}")

        # print(f"지지저항_ 현재종목코드 = {code}, 현재가격 = {current_price}, 이전가격 = {previous_price}, 크로스가격 = {self.crossPrice[code]}, 지지리스트 ={target_prices_asc}")
        # 가격 상승/하락 여부 판단
        is_uptrend = current_price > previous_price
        # 상승 또는 하락에 맞게 가격 리스트 정렬
        if is_uptrend:
            # 상승 시: 현재 가격보다 높은 목표 가격만 고려
            # print("지지저항_ 상승중")
            try:
                target_prices_asc = sorted([price for price in target_prices if price >= self.crossPrice[code]])
            except Exception as e:
                print(f" target_prices_asc Error in support_resistance: {e}")

            # print("상승 추세 감지: 상승 매매 전략 수행")
            # print(f"가격 ^^^ 상승 중.. 현재종목코드 = {code}, 현재가격 = {current_price}, 이전가격 = {previous_price}, 크로스가격 = {self.crossPrice[code]}, 지지리스트 ={target_prices_asc}")
            if len(target_prices_asc) > 1:
                if current_price > target_prices_asc[1]:    # 항상 현재가격 아래 지지선부터 설정하기 때문에...1 ->  target_prices_asc[1]
                    # self.crossPrice[code][0] = target_prices_asc[1]
                    self.crossPrice[code] = target_prices_asc[1]
                    print(f"지금 돌파한 가격 = { self.crossPrice[code]}")
        else:
            # 하락 시: 현재 가격보다 낮은 목표 가격만 고려
            # target_prices_desc = sorted([price for price in target_prices if price < current_price], reverse=True)
            try:
                # print(f"가격 vvv 하락 중.. 현재종목코드 = {code}, 현재가격 = {current_price}, 이전가격 = {previous_price}, 크로스가격 = {self.crossPrice[code]},")
                pass
            except Exception as e:
                print(f" 하락 출력 Error in support_resistance: {e}")

            if code not in self.buy_prices:
                self.buy_prices[code] = []  # 초기값으로 빈 리스트 할당
            # print(f" 하락매매에서  self.buy_prices[code] => {self.buy_prices[code]}, len(self.buy_prices[code]) => {len(self.buy_prices[code])}")
            # 하락 매매 전략
            try:
                if len(self.buy_prices[code]) < 3:
                    # current_price -= increment  # 현재 가격 하락
                    # print(f"self.buy_prices[code] => {self.buy_prices[code]}, len(self.buy_prices[code]) => {len(self.buy_prices[code])}")
                    if len(self.buy_prices[code]) == 0:
                        target_prices_desc = sorted([price for price in target_prices if price <= self.crossPrice[code]], reverse=True)
                        # print(f" len(self.buy_prices[code]) == 0, 현재가 = {current_price}, 리스트 ={target_prices_desc}")
                    elif len(self.buy_prices[code]) == 1:
                        target_prices_desc = sorted([price for price in target_prices if price < self.buy_prices[code][0]], reverse=True)
                        # print(f" len(self.buy_prices[code]) == 1,현재가 = {current_price}, 리스트 ={target_prices_desc}")
                    elif len(self.buy_prices[code]) == 2:
                        target_prices_desc = sorted([price for price in target_prices if price < self.buy_prices[code][1]], reverse=True)
                        # print(f" len(self.buy_prices[code]) == 2,현재가 = {current_price}, 리스트 ={target_prices_desc}")

                    # print(f" 하락 지지리스트 = {target_prices_desc}")
                    for price in target_prices_desc:
                        # print(f"$가격비교 중... 현재종목코드 = {code}, 현재가격 = {current_price}, 지지가격 = {price}, 매수한가격 = {self.buy_prices[code]}, 지지리스트 = {target_prices_desc}")

                        if current_price <= price and price not in self.buy_prices[code]:  # 목표가격에 도달 또는 돌파 시 매수
                            # if self.매수Lock[code] == None and self.매도Lock[code] == None:
                            # 매수Lock 및 매도Lock에서 None 체크
                            # self.buy_prices[code].append(price)
                            # self.crossPrice[code] = price
                            # print(f"$ 매수 조건... 가격비교 중... 현재종목코드 = {code}, 현재가격 = {current_price}, 지지가격 = {price}, 매수한가격 = {self.buy_prices[code]}, 지지리스트 = {target_prices_desc}")

                            if self.매수Lock.get(code) is None and self.매도Lock.get(code) is None:
                                # 현재 매수, 매도가 진행되지 않고 있을때.. 주문가능,
                                self.매수Lock[code] = 'B'
                                self.buy_prices[code].append(price)
                                # self.crossPrice[code][0] = price
                                self.crossPrice[code] = price

                                # 매수 주문 후 매수 완료가 될때 까지 리얼데이터 임시 정지..
                                try:
                                    if code in self.kospi_codes:
                                        if type(self.XR_S3_) is not type(None):
                                            self.XR_S3_.UnadviseRealDataWithKey(종목코드=code)
                                            print(" S3 코스피 리얼데이터 종목 임시 정지..", code)
                                    if code in self.kosdaq_codes:
                                        if type(self.XR_K3_) is not type(None):
                                            self.XR_K3_.UnadviseRealDataWithKey(종목코드=code)
                                            print(" K3 코스닥 리얼데이터 종목 임시 정지...", code)
                                except Exception as e:
                                    print(f"리얼데이터 정지... 오류: {e}")

                                # 매수 지지선 매수
                                try:
                                    # 매수 주문 처리 로직
                                    self.QA_CSPAT00600.Query(계좌번호=self.계좌번호, 입력비밀번호=self.비밀번호,
                                                             종목번호=code, 주문수량=str(현매수량),
                                                             주문가=str(price), 매매구분=self.매수,
                                                             호가유형코드=self.매수방법, 신용거래코드=self.신용거래코드,
                                                             주문조건구분=self.조건없음)
                                    print(f"$$ 매수 주문 완료: 종목코드 = {code}, 주문가격 = {price}")
                                except Exception as e:
                                    print(f"매수 주문 오류: {e}")


                                # ToTelegram(__class__.__name__ + "매수주문 : %s %s 주문수량:%s 주문가격:%s" % (단축코드, 종목명, 수량, 현재가))

                                # print(f"매수: 현재가격 = {current_price}, 목표가격 = {price}, 돌파가격 = {price * 1.01}")
                                data = [self.Name, self.UUID, ydm, code, " ", '매수', price, 0, 현매수량, dd, 0]
                                self.주문기록(data=data)

                                print(f"$$매수가격 ..현재종목코드 = {code}, 현재가격 = {current_price},현재매주주문가격 = {price}, 매수한가격들 = {self.buy_prices[code]}, 크로스가격 = {self.crossPrice[code]}")
                                # try:
                                #     print(" 하락 매매전략 진행 CSPAT00600 TR 요청...: ", code, data)
                                # except:
                                #     print("* 프린터 에러 발생 라인:415 ")

                                break  # 한 번에 한 가격만 매수

            except Exception as e:
                print(f" len(self.buy_prices[code])  Error in support_resistance: {e}")

        # 결과 출력
        # print("최종 매수한 가격들:", buy_prices)


    def OnReceiveMessage(self, systemError, messageCode, message):
        일자 = "{:%Y-%m-%d %H:%M:%S.%f}".format(datetime.datetime.now())
        클래스이름 = self.__class__.__name__
        print(일자, 클래스이름, systemError, messageCode, message)

    def OnReceiveData(self, szTrCode, result):
        #계좌정보조회
        self.테스트용2 = False
        if szTrCode == 't0424':
            df1, df2 = result
            # print(" t0424----------------", len(df2))
            if len(df2) > 0:
                for i in range(len(df2)):
                    종목코드 = df2["종목번호"].values[i].strip()
                    수량 = df2["매도가능수량"].values[i]
                    매수가 = df2["평균단가"].values[i]
                    매입금액 = df2["매입금액"].values[i]
                    종목명 = df2["종목명"].values[i].strip()
                    매도가능수량 = df2["매도가능수량"].values[i]

                    if 매도가능수량 > 0:
                        self.portfolio[종목코드] = CPortStock(종목코드=종목코드, 종목명=종목명, 매수가=매수가, 수량=수량, 누적수량=수량, 매수일='')
                        print("토프폴리오에 추가", self.portfolio[종목코드].종목코드, self.portfolio[종목코드].종목명, self.portfolio[종목코드].매수가,
                              self.portfolio[종목코드].누적수량, self.portfolio[종목코드].매수일)
                    else:
                        self.금일매도종목.append(종목코드)
                        print("매도한 종목은 금일매도종목에 추가", self.금일매도종목)


                CTS_종목번호 = df1['CTS_종목번호'].values[0].strip()
                if CTS_종목번호 != '':
                    self.XQ_t0424.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 단가구분='1', 체결구분='0', 단일가구분='0', 제비용포함여부='1', CTS_종목번호=CTS_종목번호)
                                                                                              #체결구분->전체 잔고 조회
                else:
                    self.주문기록읽기() #금일 주문 기록 읽거 중복 매매 방지를 위해... 한번 읽는다.
                    오늘 = "{:%Y-%m-%d}".format(datetime.datetime.now())
                    print(" 현재 로봇 기존 종목 정보 읽기 : ", self.myOrderInf)
                    for k, v in self.myOrderInf.items():
                        #todo:중복 매수를 피하기 위해 금일 매수 안하도록 할것
                        print("  주문한 정보: ", v.매수일, v.매매구분 , v.단축종목번호)
                        code = v.단축종목번호
                        if v.매수일 == 오늘 :
                            self.금일매도종목.append(code)
                            print(" 주의) 오늘 이 종목들은 매매하지 않을것!!! ->", self.금일매도종목)

                            # 중목매매방지를 위해 리얼 데이터 종목에서 제거
                            if code in self.kospi_codes:
                                if type(self.XR_S3_) is not type(None):
                                    self.XR_S3_.UnadviseRealDataWithKey(종목코드=code)
                                    print("     t0424 코스피 UnadviseRealDataWithKey ")
                            if code in self.kosdaq_codes:
                                if type(self.XR_K3_) is not type(None):
                                    self.XR_K3_.UnadviseRealDataWithKey(종목코드=code)
                                    print("     t0424 코스닥 UnadviseRealDataWithKey ")

                    # print(" 3확인용: ")

                    #todo: 주문기록 삭제시에는 반드시 종목번호가 있는지 확인하고 삭제한다.  오류로 프로그램 꺼짐. 주의
                    # *****주문 종목 확인 구별 시 uuid로 하지 않고 그냥 로봇 추적법으로 한다. 이유는 uuid 변경되도 종목은 유지되어야 하므로...
                    # 단축종목번호 = '006140'
                    # self.주문기록삭제(단축종목번호)
                    # self.주문기록읽기()
                    # print("주문기록삭제후 남은종목", self.myOrderInf)

            # todo: 실시간 조건검색 실행 ***************************************************************************************************
            print("...조건검색 요청..t1857쿼리에서 실시간으로 할려면 1,  실시간 안할려면 0을 설정...........")
            self.XQ_t1857 = t1857(parent=self, 식별자=uuid.uuid4().hex)
            self.XQ_t1857.Query(실시간구분='1', 종목검색구분='F', 종목검색입력값=self.ACF파일)
                                #### 실시간 구분 1으로 설정, OnReceiveSearchRealData  작용.
                                #매매디버깅
        # 미체결 수량확인
        if szTrCode == 't0425':
            print("t0425 미체결 수량 확인 시작--------------------------------------------------")
            df, df1 = result
            if len(df1) > 0:
                try:
                    print(" 1 미체결 종목 리스트", df1)
                except:
                    print("* 프린터 에러 발생 라인:254 ")

                for i in range(len(df1)):
                    종목번호 = df1['종목번호'].values[i].strip()            #str
                    주문수량 = df1['주문수량'].values[i]                    #int
                    원주문번호 = df1['주문번호'].values[i]                  #int
                    구분 = df1['구분'].values[i].strip()                  #str
                    if 구분 == '매수':
                        time.sleep(1)
                        try:
                            print(" 강제 취소주문 : 종목번호, 주문수량, 원주문번호", 종목번호, 주문수량, 원주문번호)
                        except:
                            print("* 프린터 에러 발생 라인:254 ")

                        self.QA_CSPAT00800.Query(원주문번호=원주문번호, 계좌번호=self.계좌번호, 입력비밀번호=self.비밀번호, 종목번호=종목번호, 주문수량=주문수량)

                #종목이 있을때 이후 추가 요청 가능
                CTS_주문번호 = df['주문번호'].values[0].strip()
                if CTS_주문번호 != '':
                    if self.미체결용 == True:
                        print(" t0425...... 미체결 일괄 전체 요청")
                        self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호='', 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호=CTS_주문번호)
                    else:
                        print(" t0425 미체결 종목만 추가 요청")
                        self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호=종목번호, 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호=CTS_주문번호)

            else:
                print("미체결 종목 없음")

        #차트인덱스
        if szTrCode == 'CHARTINDEX':
            # print("CHARTINDEX 시작--------------------------------------------------")
            식별자, 지표ID, 레코드갯수, 유효데이터컬럼갯수, df = result
            dt =  int(df['일자'].values[1]) - 1

            if self.myChartCnt[self.현재종목코드] > 2:
                code = self.현재종목코드
                # length = len(self.차트인덱스종목리스트.keys())

                flag = 'N'
                if type(code) == str:
                    # print("0....", type(code), flag)
                    # print(self.kospi_codes)
                    if code in self.kospi_codes and flag in ['N','R']:

                        # print("1....")
                        if type(self.XR_S3_) is not type(None):
                            # print("     코스피 코드 실시간 요청...1: %s " % code)
                            self.XR_S3_.AdviseRealData(종목코드=code)
                    if code in self.kospi_codes and flag in ['O']:
                        # print("2....")
                        if type(self.XR_S3_) is not type(None):
                            # print("3....")
                            if code not in self.portfolio.keys() and code not in self.매수Lock.keys() and code not in self.매도Lock.keys():
                                # print("     코스피 코드 실시간 취소...1: %s " % code)
                                self.XR_S3_.UnadviseRealDataWithKey(종목코드=code)
                    if code in self.kosdaq_codes and flag in ['N','R']:
                        # print("4....")
                        if type(self.XR_K3_) is not type(None):
                            # print("     코스닥 코드 실시간요청...2 : %s" % code)
                            self.XR_K3_.AdviseRealData(종목코드=code)
                    if code in self.kosdaq_codes and flag in ['O']:
                        # print("5....")
                        if type(self.XR_K3_) is not type(None):
                            print("6....")
                            if code not in self.portfolio.keys() and code not in self.매수Lock.keys() and code not in self.매도Lock.keys():
                                # print("     코스닥 코드 실시간 취소...2: %s " % code)
                                self.XR_K3_.UnadviseRealDataWithKey(종목코드=code)

                # 현재 가지고 있는 포트폴리오의 실시간데이타를 받는다.
                for code in self.portfolio.keys():
                    if code in self.kospi_codes:
                        if type(self.XR_S3_) is not type(None):
                            self.XR_S3_.AdviseRealData(종목코드=code)
                    if code in self.kosdaq_codes:
                        if type(self.XR_K3_) is not type(None):
                            self.XR_K3_.AdviseRealData(종목코드=code)

                time.sleep(1)
                del self.myChartCnt[self.현재종목코드]
                numlist = len(self.기준봉검색종목)
                if numlist > 0 :
                    self.현재종목코드 = self.기준봉검색종목.pop(0)
                    self.차트인덱스종목리스트[self.현재종목코드] = [0.0]
                    self.myChartCnt[self.현재종목코드] = 1
                    self.XQ_ChartIndex.Query(지표ID='', 지표명='추적법', 지표조건설정='', 시장구분='1', 주기구분='2', 단축코드=self.현재종목코드, 요청건수=500,
                                             단위='3', 시작일자='', 종료일자='', 수정주가반영여부='1', 갭보정여부='1', 실시간데이터수신자동등록여부='0')

                else:
                    print("~~~~~~~~~~~~~~~~~~ 차트인덱스 검색 종료.... 매매 시작!!!!!!!! ")

            else:
                # todo 추적법 가격 검색 방법 함수 구현
                # print(df)
                self.차트인덱스종목리스트[self.현재종목코드].extend(list(set(df['지표값1'])))
                self.차트인덱스종목리스트[self.현재종목코드].extend(list(set(df['지표값2'])))
                self.차트인덱스종목리스트[self.현재종목코드].sort()

                #지지선 1% 오차 정리
                # print("0전", self.차트인덱스종목리스트[self.현재종목코드])
                self.차트인덱스종목리스트[self.현재종목코드] = [item for item in self.차트인덱스종목리스트[self.현재종목코드] if item != 0 and item != 0]
                # print("0 제거", self.차트인덱스종목리스트[self.현재종목코드])
                for i in range(1, len(self.차트인덱스종목리스트[self.현재종목코드]) - 1):
                    if(i < len(self.차트인덱스종목리스트[self.현재종목코드]) - 1):
                        gap = (self.차트인덱스종목리스트[self.현재종목코드][i + 1] / self.차트인덱스종목리스트[self.현재종목코드][i]) - 1
                        # print("3전", i, self.차트인덱스종목리스트[self.현재종목코드][i], self.차트인덱스종목리스트[self.현재종목코드][i+ 1])
                        if gap < 0.005:
                            # self.차트인덱스종목리스트[self.현재종목코드][i+1] = (self.차트인덱스종목리스트[self.현재종목코드][i + 1] + self.차트인덱스종목리스트[self.현재종목코드][i]) / 2
                            self.차트인덱스종목리스트[self.현재종목코드][i+1] = self.차트인덱스종목리스트[self.현재종목코드][i]
                            self.차트인덱스종목리스트[self.현재종목코드].pop(i)

                print(" 0 제거 후 지지선 정리", self.차트인덱스종목리스트[self.현재종목코드])

                # try:
                #     # print(self.차트인덱스종목리스트)
                #     pass
                # except:
                #     print("* 프린터 에러 발생 라인:349 ")

                self.myChartCnt[self.현재종목코드] += 1
                time.sleep(2)
                self.XQ_ChartIndex.Query(지표ID='', 지표명='추적법', 지표조건설정='', 시장구분='1', 주기구분='2', 단축코드=self.현재종목코드, 요청건수=500,
                                         단위='3', 시작일자='', 종료일자= str(dt), 수정주가반영여부='1', 갭보정여부='1', 실시간데이터수신자동등록여부='0')

            # print("CHARTINDEX 끝--------------------------------------------------")

        # 종목검색
        if szTrCode == 't1857':
            print("t1857 시작--------------------------------------------------")
            if self.running:
                식별자, 검색종목수, 포착시간, 실시간키, df = result
                self.종목수인덱스 = 0
                self.기준봉종목수 = 검색종목수
                try:
                    print("확인0", df)
                except:
                    print("* 프린터 에러 발생 라인:366 ")

                # todo 테스트용**************************************************************************************
                # 단축종목번호 = '108860'
                # 현재시각 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # 매수일 = "{:%Y-%m-%d}".format(datetime.datetime.now())
                # data = [self.Name, self.UUID, 현재시각, 단축종목번호, '', '매수', 1, 2, 3, 매수일, 1]
                # print("주문 날짜", data)
                # self.주문기록(data=data)
                # print("주문기록")

                self.주문기록읽기()  # 금일 주문 기록 읽거 중복 매매 방지를 위해... 한번 읽는다.
                try:
                    # print(" t1857 실시간 조건검색에서 현재 로봇 기존 종목 정보 읽기 : ", list(self.myOrderInf.keys()))
                    pass
                except:
                    print("* 프린터 에러 발생 라인:381 ")

                if 식별자 == self.XQ_t1857.식별자:
                    for idx, row in df[['종목코드','종목상태']].iterrows():
                        #idx row 값이 self.기준봉검색장목, flag 에 저장된다. 조건 검색한 종목 한번만 불러옴.
                        code, flag = row
                        # print("code",type(code))
                        self.기준봉검색종목.insert(self.종목수인덱스, code)
                        try:
                            # print("조건검색종목리스트:", self.기준봉검색종목[self.종목수인덱스], self.종목수인덱스, flag)
                            pass
                        except:
                            print("* 프린터 에러 발생 라인:391 ")
                        self.종목수인덱스 = self.종목수인덱스 + 1

                    if not list(self.myOrderInf) :
                        print("리스트 비어있다.4")
                        pass
                    else:
                        print(" 오늘 거래한 종목 리스트가 있다.")
                        for k, v in self.myOrderInf.items():
                            try:
                                # print("  4 조건 검색에서 제거 전 종목 리스트... ",  self.기준봉검색종목)
                                pass
                            except:
                                print("* 프린터 에러 발생 라인:403 ")
                            if v.단축종목번호 in self.기준봉검색종목:
                                self.기준봉검색종목.remove(v.단축종목번호)
                                try:
                                    # print("  5 조건 검색에서 제거 후 종목 리스트... ",  self.기준봉검색종목)
                                    pass
                                except:
                                    print("* 프린터 에러 발생 라인:409 ")

                    try:
                        # print(type("755line ", self.buy_prices[self.현재종목코드]))
                        pass
                    except:
                        print("* 프린터 에러 발생 라인:413 ")
                    # 리스트 첫번째 코드 가져온다.
                    self.현재종목코드 = self.기준봉검색종목.pop(0)
                    # self.현재종목코드 = str(218150)   #매매디버깅

                    # print(type("755line ", self.buy_prices[self.현재종목코드]), len(self.buy_prices[self.현재종목코드]))
                    self.가우시안가격리스트[self.현재종목코드] = [0.0]

                    # todo : 건수= 값으로 일별 데이터를 불러올 수 있다.
                    print(self.현재종목코드)
                    self.XQ = t1305(parent=self)
                    self.XQ.Query(단축코드=self.현재종목코드, 일주월구분='1', 날짜='', IDX='', 건수=300, 연속조회=False)

            # print("t1857 끝--------------------------------------------------------")


        # 체결
        if szTrCode == 'CSPAT00600':
            print("CSPAT00600 시작--------------------------------------------------")
            df, df1 = result
            if len(df1) > 0:
                주문번호 = df1['주문번호'].values[0]
                code = df['종목번호'].values[0]
                수량 = df['주문수량'].values[0]

                if 주문번호 != '0':
                    # 주문번호처리
                    self.주문번호리스트.append(str(주문번호))
                    try:
                        print("  1 주문번호리스트", self.주문번호리스트)
                        # pass
                    except:
                        print("* 프린터 에러 발생 라인:441 ")

                    self.주문번호관리리스트[주문번호] = dict(종목번호= code, 주문수량= 수량)      # by david
                    try:
                        print("  2 주문번호관리리스트", self.주문번호관리리스트)
                    except:
                        print("* 프린터 에러 발생 라인:447 ")

            # 현재 가지고 있는 포트폴리오의 실시간데이타를 받는다.
            if code in self.portfolio.keys():
                try:
                    if code in self.kospi_codes:
                        if type(self.XR_S3_) is not type(None):
                            self.XR_S3_.AdviseRealData(종목코드=code)
                            print(" 코스피 CSPAT00600 리얼데이터 종목 임시 정지..", code)
                    if code in self.kosdaq_codes:
                        if type(self.XR_K3_) is not type(None):
                            self.XR_K3_.AdviseRealData(종목코드=code)
                            print(" 코스닥 CSPAT00600 리얼데이터 종목 임시 정지..", code)
                except Exception as e:
                    print(f"리얼데이터 정지... 오류: {e}")

            # print("CSPAT00600 끝--------------------------------------------------------")


        # 취소주문
        if szTrCode == 'CSPAT00800':
            print("CSPAT00800 시작--------------------------------------------------")
            df, df1 = result
            try:
                print(df)
            except:
                print("* 프린터 에러 발생 라인:459 ")

            print("CSPAT00800 끝--------------------------------------------------------")

        # 기준봉 매매 종목 선정 , 일봉 데이터
        if szTrCode == 't1305':
            print("t1305 시작- 일봉데이터 요청-------------------------------------------------------")
            CNT, 날짜, IDX, df = result
            # print("     일자 데이터 ", CNT, 날짜, IDX)
            # print(df)

            매매선정 = False
            code = self.현재종목코드
            flag = 'N'
            print("     t1305의 현재종목코드 :" ,code, type(code), type(self.현재종목코드))

            nowOpen = df['시가'].values[0]
            nowHigh = df['고가'].values[0]
            nowClose = df['종가'].values[0]
            print(nowOpen,nowHigh,nowClose)

            #todo 해당 종목 기준봉 찾기 함수
            dayCnt = myReferenceRod(code,CNT, df, nowOpen, nowHigh, nowClose)
            print("     기준봉 조건 맞나? 기준봉 몇일전? ", dayCnt)

            #todo : 테스트용
            # dayCnt = 1      # 매매디버깅용
            # print("테스트용... Boxflag == True 실행... 테스트 완료 후 주석치할 것...")

            if dayCnt > 0 :
                # if dayCnt == 1 :
                #     print(df['날짜'].values[dayCnt])
                #     self.XQ_t8412.Query(단축코드=str(code), 단위=3, 요청건수=500, 조회영업일수='1', 시작일자='', 시작시간='',
                #                         종료일자=df['날짜'].values[dayCnt], 종료시간='', 연속일자='', 연속시간='', 압축여부='N', 연속조회=False)
                # elif dayCnt == 2 :
                #     print(df['날짜'].values[dayCnt - 1])
                #     self.XQ_t8412.Query(단축코드=str(code), 단위=3, 요청건수=500, 조회영업일수='2', 시작일자='', 시작시간='',
                #                         종료일자=df['날짜'].values[dayCnt-1], 종료시간='', 연속일자='', 연속시간='', 압축여부='N', 연속조회=False)
                # else:
                #     print(df['날짜'].values[dayCnt - 2])
                #     self.XQ_t8412.Query(단축코드=str(code), 단위=3, 요청건수=500, 조회영업일수='3', 시작일자='', 시작시간='',
                #                         종료일자=df['날짜'].values[dayCnt-2], 종료시간='', 연속일자='', 연속시간='', 압축여부='N', 연속조회=False)

                print(df['날짜'].values[dayCnt])
                self.XQ_t8412.Query(단축코드=str(code), 단위=3, 요청건수=500, 조회영업일수='1', 시작일자='', 시작시간='',
                                    종료일자=df['날짜'].values[dayCnt], 종료시간='', 연속일자='', 연속시간='', 압축여부='N', 연속조회=False)

            else:
                # if self.종목수인덱스 >= 0:
                numlist = len(self.기준봉검색종목)

                numlist = 0         # 매매디버깅용
                # print("************ 한종목만 테스트 하기 위해 numlist를 0으로 세팅...실제 구동 시 주석처리 할것!!!!********************" )

                if numlist > 0 :
                # if numlist > 27:
                #     print("중요.******** 한종목만 테스트 하기 위해 numlist > 27 세팅...실제 구동 시 주석처리 할것!!!!********************")
                    self.현재종목코드 = self.기준봉검색종목.pop()
                    # self.crossPrice[self.현재종목코드] = 0
                    # self.buy_prices[self.현재종목코드].append(0)
                    # self.myN분봉Cnt[self.현재종목코드] = 1
                    # self.차트인덱스종목리스트[self.현재종목코드] = [0.0]
                    self.가우시안가격리스트[self.현재종목코드] =[0.0]
                    print("\n\n     이번에 t1305에서 쿼러할 종목코드, 남은종목수:", self.현재종목코드, numlist)

                    time.sleep(1)
                    self.XQ = t1305(parent=self)
                    self.XQ.Query(단축코드=self.현재종목코드, 일주월구분='1', 날짜='', IDX='', 건수=300, 연속조회=False)

            print("t1305 끝----------------------------------------------------")

        # 분봉 데이터
        if szTrCode == 't1302':
            print("  t1302 분봉 데이터 요청-")
            시간CTS, df = result
            print(df)

            print("분봉 버퍼인덱스:", self.테스트용, self.테스트용-1)
            self.minuteTime = df['시간'].values[self.테스트용-1]
            self.siga = df['시가'].values[self.테스트용-1]
            self.jongga = df['종가'].values[self.테스트용-1]
            self.staticrate = df['등락율'].values[self.테스트용-1]

            print("결과", self.minuteTime, self.siga, self.jongga, self.staticrate)

            # print("  2 끝----------------------------------------------------")
            self.Qwaitting = True

        # 주식차트 N분봉 데이터
        if szTrCode == 't8412':
            # print("t8412 시작--------------------------------------------------")
            print("  t8412 주식차트 N분봉 데이터")
            block, df = result
            code = str(self.현재종목코드)
            # print(type(self.현재종목코드))
            # length = len(self.차트인덱스종목리스트.keys())

            flag = 'N'      # 종목 실시간 진입 상태를 신규상태로 초기화
#todo gaussian 알고리즘 시작................................................................

            # totaldf = pd.concat([mydf0, mydf1, mydf2], ignore_index=True)
            totaldf = pd.concat([df], ignore_index=True)

            ## Input parameters
            slice_ = slice(0, 1500)
            peaks_range = [5, 15]
            num_peaks = -999

            sample_df = totaldf.iloc[slice_]
            sample = totaldf.iloc[slice_][["종가"]].to_numpy().flatten()
            sample_original = sample.copy()

            maxima = argrelextrema(sample, np.greater)
            minima = argrelextrema(sample, np.less)

            extrema = np.concatenate((maxima, minima), axis=1)[0]
            extrema_prices = np.concatenate((sample[maxima], sample[minima]))
            if len(extrema_prices) > 0:
                interval = extrema_prices[0] / 10000
            else:
                # extrema_prices가 비어 있을 때 처리할 코드
                interval = 0  # 또는 다른 기본값을 설정할 수 있음
            # interval = extrema_prices[0] / 10000

            # bandwidth 설정
            # bandwidth = interval
            bandwidth = interval if interval > 0 else 0.1  # 최소 bandwidth를 0.1로 설정

            price_range = None
            peaks = []  # peaks 변수를 초기화

            # print('extrema', extrema_prices)
            while num_peaks < peaks_range[0] or num_peaks > peaks_range[1]:

                if extrema_prices.size > 0:
                    # kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))
                    # initial_price = extrema_prices[0]
                    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))
                else:
                    print("KDE를 맞출 수 있는 extrema 가격이 없습니다., 이번 t1302 실행을 멈춥니다.")
                    flag = 'O'      # 지지선을 찾이 못하면 이 종목은 제거 시킴.
                    break

                a, b = min(extrema_prices), max(extrema_prices)
                price_range = np.linspace(a, b, 1000).reshape(-1, 1)
                pdf = np.exp(kde.score_samples(price_range))
                peaks = find_peaks(pdf)[0]

                num_peaks = len(peaks)
                bandwidth += interval

                if bandwidth > 100 * interval:
                    print("Failed to converge, stopping...")
                    flag = 'O'
                    break

            # print('rnage',price_range)
            if price_range is not None and peaks.size > 0:
                # Convert price_range[self.peaks] to integers and store them in the appropriate dictionary entry
                peak_prices = (price_range[peaks].flatten()).astype(int).tolist()
                self.가우시안가격리스트[self.현재종목코드] = peak_prices
                # print("전 self.가우시안가격리스트[%s]" % self.현재종목코드, self.가우시안가격리스트[self.현재종목코드])
                self.가우시안가격리스트[self.현재종목코드] = [
                    self.UpPrice(p)  # 호가로 변경
                    for p in self.가우시안가격리스트[self.현재종목코드]
                ]
                print("후 self.가우시안가격리스트[%s]" % self.현재종목코드, self.가우시안가격리스트[self.현재종목코드] )
            else:
                print("peaks가 정의되지 않거나 price_range가 None입니다.")
                flag = 'O'

            # draw_candle_chart(sample_df, self.price_range[self.peaks], region=0.0001)
            # draw_line_chart(sample_df, self.price_range[self.peaks], region=0.0001, mavg=3)
            # plt.show()
#todo gaussian 알고리즘 끝...................................................................

            if type(code) == str:
                # print("0....", type(code), flag)
                # print(self.kospi_codes)
                if code in self.kospi_codes and flag in ['N','R']:
                    # print("1....")
                    if type(self.XR_S3_) is not type(None):
                        print("     코스피 코드 실시간 요청...1: %s " % code)
                        self.XR_S3_.AdviseRealData(종목코드=code)
                if code in self.kospi_codes and flag in ['O']:
                    # print("2....")
                    if type(self.XR_S3_) is not type(None):
                        # print("3....")
                        if code not in self.portfolio.keys() and code not in self.매수Lock.keys() and code not in self.매도Lock.keys():
                            # print("     코스피 코드 실시간 취소...1: %s " % code)
                            self.XR_S3_.UnadviseRealDataWithKey(종목코드=code)
                if code in self.kosdaq_codes and flag in ['N','R']:
                    # print("4....")
                    if type(self.XR_K3_) is not type(None):
                        print("     코스닥 코드 실시간요청...2 : %s" % code)
                        self.XR_K3_.AdviseRealData(종목코드=code)
                if code in self.kosdaq_codes and flag in ['O']:
                    # print("5....")
                    if type(self.XR_K3_) is not type(None):
                        print("6....")
                        if code not in self.portfolio.keys() and code not in self.매수Lock.keys() and code not in self.매도Lock.keys():
                            # print("     코스닥 코드 실시간 취소...2: %s " % code)
                            self.XR_K3_.UnadviseRealDataWithKey(종목코드=code)

            # 현재 가지고 있는 포트폴리오의 실시간데이타를 받는다.
            for code in self.portfolio.keys():
                if code in self.kospi_codes:
                    if type(self.XR_S3_) is not type(None):
                        self.XR_S3_.AdviseRealData(종목코드=code)
                if code in self.kosdaq_codes:
                    if type(self.XR_K3_) is not type(None):
                        self.XR_K3_.AdviseRealData(종목코드=code)

            time.sleep(1)
            # del self.myN분봉Cnt[self.현재종목코드]
            numlist = len(self.기준봉검색종목)
            # numlist = 0 # 매매디버깅
            if numlist > 0 :
                self.현재종목코드 = self.기준봉검색종목.pop(0)
                self.가우시안가격리스트[self.현재종목코드] = [0.0]
                print("  t8412에서 다음 t1305 쿼리요청", self.현재종목코드, numlist)
                self.XQ = t1305(parent=self)
                self.XQ.Query(단축코드=self.현재종목코드, 일주월구분='1', 날짜='', IDX='', 건수=300, 연속조회=False)

            else:
                # self.가우시안가격리스트[146320][12160, 12340, 13780, 14000, 14100, 14590, 14790, 14950, 15250, 15360, 15470, 15780, 15870]
                # 현재가_list = [14150, 14070, 14060, 14110, 14400, 14600, 14700, 14550, 14200, 14050]
                # self.이전가격[code] = 0
                #
                # for 현재가 in 현재가_list:
                #     self.support_resistance(code, int(현재가), int(self.이전가격[code]))
                #     self.이전가격[code] = int(현재가)

                # 매매디버깅
                # nowDate = datetime.datetime.now()
                # self.VIsticList[code] = dict(firstSetTime=nowDate, lastTime=nowDate, lastValue=0,
                #                              currentRate=0.0,
                #                              firstTimeFlag=0, siga=0, jong=0, cnt=0, upPrice=0, downPrice1=0,
                #                              downPrice2=0)
                # print(f"(종목분봉, self.VIsticList[code] = {self.VIsticList[code]} )")

                print("~~~~~~~~~~~~~~~~~~t8412 주식차트 N분봉 데이터 검색 종료.... 매매 시작!!!!!!!! ")

            print("t8412 끝--------------------------------------------------")
            # self.Qwaitting = True


    def OnReceiveSearchRealData(self, szTrCode, lst):
        식별자, result = lst
        # try:
        #     print("OnReceiveSearchRealData 함수 실행-------------------------", 식별자, result)
        # except:
        #     print("* 프린터 에러 발생 라인:483 ")

        if 식별자 == self.XQ_t1857.식별자:
            try:
                code = result['종목코드']
                flag = result['종목상태']
                codePrice = result['현재가']

                # 테스트용
                # code = str(146320)
                # flag = 'N'

                # codePrice = result['현재가']
                # print("     여기...2", code,flag)
                sourchCode = False

                if flag in ['N']:
                    try:
                        nowDate = datetime.datetime.now()
                        self.VIsticList[code] = dict(firstSetTime = nowDate, lastTime = nowDate, lastValue = codePrice, currentRate = 0.0,
                                                     firstTimeFlag = 0, siga = 0, jong = 0, cnt = 0, upPrice = 0, downPrice1 = 0, downPrice2 = 0)

                        # self.VIsticList[code] = dict(firstSetTime = nowDate, lastTime = nowDate, lastValue = codePrice, currentRate = 0.0,
                        #                              firstTimeFlag = 0, siga = 0, jong = 0, cnt = 0)

                        print("종목분봉", self.VIsticList[code])
                        print("  실시간 신규종목 N ... ", code)
                    except:
                        print("* 프린터 에러 발생 라인:496 ")

                    sourchCode = True
                    self.주문기록읽기()  # 금일 주문 기록 읽거 중복 매매 방지를 위해... 한번 읽는다.
                    # # print(" 실시간 조건검색에서 현재 로봇 기존 종목 정보 읽기 : ", self.myOrderInf)

                    if not list(self.myOrderInf) :
                        print(" 1매매 이력 리스트가 비어있다.")
                        # pass
                    else:
                        for k, v in self.myOrderInf.items():
                            # todo:중복 매수를 피하기 위해 금일 매수 안하도록 할것
                            # print("  주문한 정보: ", v.매수일, v.매매구분, v.단축종목번호)
                            if code == v.단축종목번호:
                                sourchCode = False
                                try:
                                    print("  금일 매매한 종목과 중복 됨... ", v.단축종목번호)
                                except:
                                    print("* 프린터 에러 발생 라인:514 ")

                                break

                if sourchCode == True:
                    if type(code) == str:
                        if code in self.kospi_codes and flag in ['N']:
                            if type(self.XR_S3_) is not type(None):
                                # self.XR_S3_.AdviseRealData(종목코드=code)
                                # 리스트 마지막에 코드 추가
                                self.기준봉검색종목.append(code)
                                self.현재종목코드 = code
                                self.myChartCnt[self.현재종목코드] = 1
                                # self.차트인덱스종목리스트[self.현재종목코드] = [0.0]
                                self.가우시안가격리스트[self.현재종목코드] =[0.0]
                                print(" 쿼리요청t1305-코스피", self.현재종목코드)
                                self.XQ = t1305(parent=self)
                                self.XQ.Query(단축코드=self.현재종목코드, 일주월구분='1', 날짜='', IDX='', 건수=300, 연속조회=False)

                        if code in self.kosdaq_codes and flag in ['N']:
                            if type(self.XR_K3_) is not type(None):
                                # self.XR_K3_.AdviseRealData(종목코드=code)
                                # 리스트 마지막에 코드 추가
                                self.기준봉검색종목.append(code)
                                self.현재종목코드 = code
                                self.myChartCnt[self.현재종목코드] = 1
                                # self.차트인덱스종목리스트[self.현재종목코드] = [0.0]
                                self.가우시안가격리스트[self.현재종목코드] = [0.0]
                                print(" 쿼리요청t1305-코스닥..", self.현재종목코드)
                                self.XQ = t1305(parent=self)
                                self.XQ.Query(단축코드=self.현재종목코드, 일주월구분='1', 날짜='', IDX='', 건수=300, 연속조회=False)

            except Exception as e:
                # print("     여기...3", code,flag)
                클래스이름 = self.__class__.__name__
                함수이름 = inspect.currentframe().f_code.co_name
                print("%s-%s %s" % (클래스이름, 함수이름, get_linenumber()), e)
            finally:
                pass

        # print("OnReceiveSearchRealData 함수 끝 ---------------------------------")

    def OnReceiveRealData(self, szTrCode, result):
        # print("온리스트리얼데이터 위치", szTrCode, result)
        if szTrCode == 'SC1':
            print('SC1 시작-----------------------------------------------------')
            print(result)

            체결시각 = result['체결시각']
            단축종목번호 = result['단축종목번호'].strip().replace('A','')
            종목명 = result['종목명']
            매매구분 = result['매매구분']
            주문번호 = result['주문번호']
            체결번호 = result['체결번호']
            주문수량 = int(result['주문수량'])
            주문가격 = int(result['주문가격'])
            체결수량 = int(result['체결수량'])
            체결가격 = int(result['체결가격'])
            주문평균체결가격 = int(result['주문평균체결가격'])
            주문계좌번호 = result['주문계좌번호']
            매도주문가능수량 = result['매도주문가능수량']

            # 내가 주문한 것이 체결된 경우 처리
            if 주문번호 in self.주문번호리스트:
                if 매매구분 == '1' or 매매구분 == 1: # 매도
                    P = self.portfolio.get(단축종목번호, None)
                    try:
                        print("  매도 데이터 누적수량", P.누적수량)
                        # pass
                    except:
                        print("* 프린터 에러 발생 라인:580 ")

                    if P != None:
                        # P.수량 = P.수량 - 체결수량
                        P.누적수량 = P.누적수량 - 체결수량
                        try:
                            # print("     111...현재 남은 매도 수량:%d, 체결수량: %d, " % (P.수량, 체결수량))
                            print("     111...현재 남은 매도 누적수량:%d, 체결수량: %d, " % (P.누적수량, 체결수량))
                            pass
                        except:
                            print("* 프린터 에러 발생 라인:587 ")

                        # if P.수량 == 0:
                        if P.누적수량 == 0:
                            #매도 완료일때 처리.
                            self.portfolio.pop(단축종목번호)
                            self.매도Lock.pop(단축종목번호)
                            self.포트폴리오종목삭제(단축종목번호)
                            self.주문번호관리리스트.pop(주문번호, None)

                            try:
                                # print(" 나머지 매수 체결안된 종목 취소 주문1.. ")
                                #해당 종목만 주문 취소 요청할 것.
                                self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호=단축종목번호, 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호='')
                                # self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호='', 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호='')
                                print("  취소주문2 ")
                            except:
                                print("* 프린터 에러 발생 라인:714 ")

                            #매도한 종목은 리얼 데이터 종목에서 제거
                            if 단축종목번호 in self.kospi_codes:
                                if type(self.XR_S3_) is not type(None):
                                    self.XR_S3_.UnadviseRealDataWithKey(종목코드=단축종목번호)
                                    print("     112...매도 코스피 UnadviseRealDataWithKey ")
                            if 단축종목번호 in self.kosdaq_codes:
                                if type(self.XR_K3_) is not type(None):
                                    self.XR_K3_.UnadviseRealDataWithKey(종목코드=단축종목번호)
                                    print("     113...매도 코스닥 UnadviseRealDataWithKey ")
                        else:
                            self.포트폴리오종목갱신(단축종목번호, P)
                            #TODO: 빠른거래시 화면의 응답속도도 영향을 주므로 일단은 커멘트처리
                            # self.parent.RobotView()
                            # ToTelegram(__class__.__name__ + "매도 : %s 체결수량:%s 체결가격:%s" % (종목명, 주문수량, 주문평균체결가격))
                    else:
                        print("     매도 주문이 없는데 매도가 들어옴")

                if 매매구분 == '2' or 매매구분 == 2: # 매수
                    P = self.portfolio.get(단축종목번호, None)
                    try:
                        print(f"    22매수 단축종목번호 ={단축종목번호}, 포트폴리오={P}")
                        # pass
                    except Exception as e:
                        print(f"* 프린터 에러 발생 라인:617 - {str(e)}")

                    try:
                        if P== None:
                            #토프폴리어 첫 등록
                            self.portfolio[단축종목번호] = CPortStock(
                                종목코드=단축종목번호,
                                종목명=종목명,
                                매수가=주문평균체결가격,
                                수량=체결수량,
                                누적수량=int(0),
                                매수일=datetime.datetime.now().strftime("%Y-%m-%d")
                            )

                            if self.portfolio[단축종목번호].수량 == 주문수량:
                                try:
                                    self.portfolio[단축종목번호].누적수량 += 주문수량
                                    self.portfolio[단축종목번호].수량 = int(0)  #수량 다시 초기화 댜음 매수를 위해...
                                    self.매수Lock.pop(단축종목번호) # 매수를 다시 할수 있도록 함.
                                    print("     1...매수 완료", self.portfolio[단축종목번호].종목코드, self.portfolio[단축종목번호].종목명)
                                except Exception as e:
                                    print(f"* 프린터 에러 발생 라인:628 - {str(e)}")

                            self.포트폴리오종목갱신(단축종목번호, self.portfolio[단축종목번호])
                            try:
                                print("토프폴리오 종목갱신", self.portfolio[단축종목번호].종목코드, self.portfolio[단축종목번호].종목명, self.portfolio[단축종목번호].매수가,
                                                          self.portfolio[단축종목번호].누적수량, self.portfolio[단축종목번호].매수일)
                                # pass
                            except:
                                print("* 프린터 에러 발생 라인:635 ")
                        else:
                            #이전 마지막 체결수량은 p.수량 + 현재 체결해서 들어온 체결수량을 더함.
                            # P.수량(이전 체결수량)  +  체렬수량(현재 체결된 수량)
                            P.수량 = P.수량 + 체결수량
                            if P.수량 == 주문수량:
                                try:
                                    P.매수가 = 주문평균체결가격
                                    P.누적수량 += P.수량
                                    P.수량 = int(0)  # 초기화 다음 매수를 위해...
                                    self.매수Lock.pop(단축종목번호) # 매수를 다시 할수 있도록 함.
                                    print("     2...매수완료", P.종목코드, P.종목명)
                                    print("     2...포트폴리오 매수평균가격", P.매수가)
                                except Exception as e:
                                    print(f"* 프린터 에러 발생 라인:645 - {str(e)}")

                                # self.parent.RobotView()
                                # ToTelegram(__class__.__name__ + "매수 : %s 체결수량:%s 체결가격:%s" % (종목명, 주문수량, 주문평균체결가격))

                            self.포트폴리오종목갱신(단축종목번호, P)
                            try:
                                print("토프폴리오 종목갱신", P.종목코드, P.종목명, P.매수가, P.수량, P.누럭수량, P.매수일)
                                # pass
                            except Exception as e:
                                print(f"* 프린터 에러 발생 라인:654 - {str(e)}")

                    except Exception as e:
                        print(f"* 에러 발생 1262- {str(e)}")

                    # 조건검색과 체결사이에 시간 간격차 때문에 등록이 안되어 있을수도 있음
                    # 체결된 종목은 실시간 가격을 받는다.
                    if 단축종목번호 in self.kospi_codes:
                        if type(self.XR_S3_) is not type(None):
                            self.XR_S3_.AdviseRealData(종목코드=단축종목번호)
                    if 단축종목번호 in self.kosdaq_codes:
                        if type(self.XR_K3_) is not type(None):
                            self.XR_K3_.AdviseRealData(종목코드=단축종목번호)

                if self.parent is not None:
                    self.parent.RobotView()

                일자 = "{:%Y-%m-%d}".format(datetime.datetime.now())
                data = [self.Name, self.UUID, 일자, 체결시각, 단축종목번호, 종목명, 매매구분, 주문번호, 체결번호, 주문수량, 주문가격, 체결수량, 체결가격, 주문평균체결가격]
                self.체결기록(data=data)

            # print('SC1 끝----------------------------------------------------')

        if szTrCode in ['K3_', 'S3_']:
            # print("K3_, S3_ 시작----------------------------------------------")
            if self.매수거래중 == True or self.매도거래중 == True:
                단축코드 = result['단축코드']
                self.realNowCode = 단축코드
                try:
                    종목명 = self.종목코드테이블.query("단축코드=='%s'" % 단축코드)['종목명'].values[0]
                except Exception as e:
                    종목명 = ''

                현재가 = result['현재가']
                self.현재가격[단축코드] = 현재가

                # 추적법 매수 수량 결정
                수량 = ((self.단위투자금) // 현재가)
                수량 = 1      #테스트용

                현재시각 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                nowMinute = datetime.datetime.now()
                매수일 = "{:%Y-%m-%d}".format(datetime.datetime.now())

                # if self.parent is not None:
                #     self.parent.statusbar.showMessage("[%s]%s %s %s" % (현재시각, 단축코드, 종목명, 현재가))

                # 서버에 있는 포트폴리오에서 종목코드가 있는지 한다. 머니봇 첫번째 실행 시 포트폴리오를 읽어와 self.portfolio 에 저장한다.
                P = self.portfolio.get(단축코드, None)
                매수락 = self.매수Lock.get(단축코드, None)
                매도락 = self.매도Lock.get(단축코드, None)

                # print("     종목명, 상태 P..." , 종목명, P)
                # print("     상태 매수락 : " , 매수락)
                # print("     상태 매도락 : " , 매도락)

                # 딕셔너리에서 키가 없으면 0을 기본값으로 설정
                # prePrice = self.이전가격.get(단축코드, 0)
                # Check if key exists and print its value
                if 단축코드 in self.이전가격:
                    # print(f"단축코드 {단축코드} exists in self.이전가격 with value: {self.이전가격[단축코드]}")
                    pass
                else:
                    self.이전가격[단축코드] = 0
                    print(f"단축코드 {단축코드} not found in self.이전가격")

                # self.support_resistance(단축코드, int(현재가), int(self.이전가격[단축코드]))
                # self.이전가격[단축코드] = 현재가
                # print(" 2K3_S3_현재 종목명, 단축코드, 현재가 :", 종목명, 단축코드, 현재가)
                # if P == None: - 사용안함.
                # #TODO: 오늘 매도했거나 매수했던 종목은 매수하지 않는다.  "금일매도종목"에 추가...
                lst = set(self.portfolio.keys()).union(self.매수Lock.keys())

                if 단축코드 not in self.금일매도종목 and 수량 > 0:
                    if self.매수거래중 == True:
                        lst = set(self.portfolio.keys()).union(self.매수Lock.keys())
                        if len(lst) <= self.포트폴리오수:
                            # vi 종목리스트에 현재코드가 있는지 확인 없으면 패스
                            # 실시간 검색에서 나오는 경우 해당 됨.
                            # print("     매수 모니터링 리스트..", self.VIsticList)
                            if 단축코드 in self.VIsticList:
                                #vi 발동 후 첫 1분동 시가 시간을 0으로 초기화 후 60초 대기 시킴.
                                checkflag = self.VIsticList[단축코드].get('firstTimeFlag')
                                if checkflag == 0:
                                    #현재lastValue는 VI 발동 시 가격 저장  이후에는 리얼 현재가 저장됨.
                                    VilastValue = self.VIsticList[단축코드].get('lastValue')

                                    self.VIsticList[단축코드]['firstTimeFlag'] = 1  # 종목 매매 종료까지 세팅 안 바꿈
                                    self.VIsticList[단축코드]['lastTime'] = nowMinute  # vi 발동 후 첫 1분동 시가 시간을 현재 시간으로 세팅
                                    self.VIsticList[단축코드]['lastValue'] = 현재가  # VI 발동 후 첫 1분봉 시가 지정
                                    self.VIsticList[단축코드]['cnt'] = 1
                                    try:
                                        print(f" VI 발동 후 첫 호가 정보, 현재시각: {str(nowMinute)}, 현재가:{str(현재가)}, 분봉 데이터 시간: {str(self.gDf['시간'].values[0])}, 가격: {str(self.gDf['시가'].values[0])}")
                                    except Exception as e:
                                        print(f" error VI 발동 후 첫 호가 정보 in support_resistance: {e}")

                                lastTime = self.VIsticList[단축코드].get('lastTime')
                                lastValue = self.VIsticList[단축코드].get('lastValue')

                                # print("모니터링 코드: ", 단축코드)
                                # 3회 매수하면 매수 기능은 하지 않는다. 이후에는 계속 매도, 손절 모니터링만 한다.
                                self.support_resistance(단축코드, int(현재가), int(self.이전가격[단축코드]))
                                self.이전가격[단축코드] = 현재가

                                # try:
                                #     print(f"후  ---모니터링 코드: len(self.buy_prices[단축코드]) = {len(self.buy_prices[단축코드])}, 매수락 = {매수락}, 매도락 = {매도락}, self.매도거래중 = {self.매도거래중} , self.매수Lock 리스트 ={self.매수Lock} ", 단축코드)
                                # except Exception as e:
                                #     print(f"* 모니터링 에러.. - {str(e)}")

                                # 즉 현재까지 매수 주문이 완료 됐는지 확인!
                                if len(self.buy_prices[단축코드]) > 0 and 매수락 == None:
                                    if self.매도거래중 == True:
                                        #매도 기능이 진행되면 아래 강제 10분 매수 취소 기능은 계속 연기된다...
                                        # self.VIsticList[단축코드]['lastTime'] = nowMinute
                                        # lastTime = self.VIsticList[단축코드].get('lastTime')

                                        if 현재가 > P.매수후고가:
                                            P.매수후고가 = 현재가

                                        # print("     매도 상태모니터링, 매도락 :...", 매도락)
                                        if 매도락 == None:
                                            # 수량 = P.수량
                                            수량 = P.누적수량
                                            매수가 = P.매수가
                                            # 매수일 = P.매수일.strftime("%Y-%m-%d %H:%M:%S")

                                            try:
                                                # print(" 수익 종목 시장가 매도:", 종목명)
                                                # print("$매도 모니터일  코드: %s, 종목명: %s, 현재수익율: %f, " % (단축코드, 종목명, (1-float(현재가 / 매수가))*100))
                                                # print("현재가:%d, 매수가:%d, 매도할 전체 수량: %d, 현재수익율 :%f " % (int(현재가), int(매수가), int(P.누적수량)),float(현재가 / 매수가))
                                                pass
                                            except:
                                                print("* 프린터 에러 발생 라인:772 ")

                                            # 포트폴리오의 수익률이 지정한 이상이면 매도
                                            if 현재가 > P.매수가 * (1 + self.profitstop):
                                                self.매도Lock[단축코드] = 'S'
                                                self.금일매도종목.append(단축코드)
                                                _현재가 = 현재가
                                                if self.매도방법 == '03':  # 00: 지정가 , 03: 시장가,
                                                    _현재가 = ''

                                                try:
                                                    print(" ->수익 종목 시장가 매도:", 종목명)
                                                    print("     $현재수익율: %f, 현재가:%d, 매수가:%d, 매도할 전체 수량: %d " % (
                                                    float(현재가 / 매수가), int(현재가), int(매수가), int(P.수량)))
                                                except:
                                                    print("* 프린터 에러 발생 라인:786 ")

                                                # 매수 주문 후 매수 완료가 될때 까지 리얼데이터 임시 정지..
                                                try:
                                                    if 단축코드 in self.kospi_codes:
                                                        if type(self.XR_S3_) is not type(None):
                                                            self.XR_S3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                            print(" S3 코스피 리얼데이터 종목 임시 정지..", 단축코드)
                                                    if 단축코드 in self.kosdaq_codes:
                                                        if type(self.XR_K3_) is not type(None):
                                                            self.XR_K3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                            print(" K3 코스닥 리얼데이터 종목 임시 정지...", 단축코드)
                                                except Exception as e:
                                                    print(f"리얼데이터 정지... 오류: {e}")


                                                self.QA_CSPAT00600.Query(계좌번호=self.계좌번호, 입력비밀번호=self.비밀번호,
                                                                         종목번호=단축코드, 주문수량=수량, 주문가=_현재가, 매매구분=self.매도,
                                                                         호가유형코드=self.매도방법)
                                                # TODO: 주문이 연속적으로 나가는 경우
                                                # 텔레그렘의 메세지 전송속도가 약 1초이기 때문에
                                                # 이베스트에서 오는 신호를 놓치는 경우가 있다.
                                                # ToTelegram(__class__.__name__ + "매도주문 : %s %s 주문수량:%s 주문가격:%s" % (단축코드, 종목명, 수량, 현재가))

                                                data = [self.Name, self.UUID, 현재시각, 단축코드, 종목명, '매도', 매수가, 현재가, 수량,
                                                        매수일, (현재가 - 매수가) * 수량]
                                                self.주문기록(data=data)

                                            # 손절
                                            if 현재가 < P.매수가 * (1 - self.losscut):
                                                self.매도Lock[단축코드] = 'S'
                                                self.금일매도종목.append(단축코드)
                                                _현재가 = 현재가
                                                if self.매도방법 == '03':
                                                    _현재가 = ''

                                                try:
                                                    print(" 손절 종목 시장가 매도:", 종목명)
                                                    print("     $현재수익율: %f, 현재가:%d, 매수가:%d, 매도할 전체 수량: %d " % (
                                                    float(현재가 / 매수가), int(현재가), int(매수가), int(P.수량)))
                                                except:
                                                    print("* 프린터 에러 발생 라인:809 ")

                                                # 매수 주문 후 매수 완료가 될때 까지 리얼데이터 임시 정지..
                                                try:
                                                    if 단축코드 in self.kospi_codes:
                                                        if type(self.XR_S3_) is not type(None):
                                                            self.XR_S3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                            print(" S3 코스피 리얼데이터 종목 임시 정지..", 단축코드)
                                                    if 단축코드 in self.kosdaq_codes:
                                                        if type(self.XR_K3_) is not type(None):
                                                            self.XR_K3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                            print(" K3 코스닥 리얼데이터 종목 임시 정지...", 단축코드)
                                                except Exception as e:
                                                    print(f"리얼데이터 정지... 오류: {e}")

                                                self.QA_CSPAT00600.Query(계좌번호=self.계좌번호, 입력비밀번호=self.비밀번호,
                                                                         종목번호=단축코드, 주문수량=수량, 주문가=_현재가, 매매구분=self.매도,
                                                                         호가유형코드=self.매도방법)
                                                # TODO: 주문이 연속적으로 나가는 경우
                                                # 텔레그렘의 메세지 전송속도가 약 1초이기 때문에
                                                # 이베스트에서 오는 신호를 놓치는 경우가 있다.
                                                # ToTelegram(__class__.__name__ + "매도주문 : %s %s 주문수량:%s 주문가격:%s" % (단축코드, 종목명, 수량, 현재가))

                                                data = [self.Name, self.UUID, 현재시각, 단축코드, 종목명, '손절', 매수가, 현재가, 수량,
                                                        매수일, (현재가 - 매수가) * 수량]
                                                self.주문기록(data=data)

                                else:
                                    # diffTime = nowMinute - lastTime
                                    deltaseconds = (nowMinute - lastTime).total_seconds()
                                    # print(
                                    #     f" 시간 체크... deltaseconds = {deltaseconds}, self.targetSeconds = {self.targetSeconds},(deltaseconds > self.targetSeconds) = {(deltaseconds > self.targetSeconds)}")
                                    if deltaseconds > self.targetSeconds:  # 10분 기준, 매도 기능을 하지 않는 기즌으로 10분 이상
                                        # vi 종목 자격 상실 종목 리스트 제거
                                        # del self.VIsticList[단축코드] ???
                                        self.VIsticList.pop(단축코드, None)

                                        # 매수lock 목록에서 제거
                                        try:
                                            if 단축코드 in self.매수Lock:
                                                self.매수Lock.pop(단축코드)
                                                self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호=단축코드, 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호='')
                                                print(" VI발동 후 120분 경과 후 리얼 데이터 목록에서 제거 및 주문취소 시킴828 종목:", 단축코드)
                                            else:
                                                print(" VI발동 후 120분 경과 매수lock 에 없음:", 단축코드)
                                        except:
                                            print("* 프린터 에러 발생 라인:841 ")

                                        # 중목매매방지를 위해 리얼 데이터 종목에서 제거
                                        try:
                                            if 단축코드 in self.kospi_codes:
                                                if type(self.XR_S3_) is not type(None):
                                                    self.XR_S3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                    print(" S3 코스피 리얼데이터 종목 제거834")
                                            if 단축코드 in self.kosdaq_codes:
                                                if type(self.XR_K3_) is not type(None):
                                                    self.XR_K3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                                    print(" K3 코스닥 리얼데이터 종목 종목 838")

                                            lst = set(self.portfolio.keys()).union(self.매수Lock.keys())
                                            print("  @현재까지 매수 요청한 포트롤리오 수 841 = ", len(lst))
                                        except:
                                            print("* 프린터 에러 발생 라인:843 ")
                else:
                    try:
                        # print(
                        #     f"금일 매도한 종목 있다 self.금일매도종목 {self.금일매도종목}, 수량 ={수량}, 매수Lock key 리스트 = {lst}, len(lst) < self.포트폴리오수 = {len(lst) < self.포트폴리오수}")
                        pass
                    except Exception as e:
                        print(f" 2K3_S3_ Error in support_resistance: {e}")

            # print("K3_, S3_ 끝-----------------------------------------------------")

    def OnClockTick(self):
        current = datetime.datetime.now()
        current_str = current.strftime('%H:%M:%S')
        # print("%s : %s" % (__class__.__name__ , current_str))

        if self.checkRobotCnt >= 300 :
            self.checkRobotCnt = 0
            try:
                print("로봇 구동 체크: %s, 현재시각:%s " % (self.Name, current_str))
            except:
                print("* 프린터 에러 발생 라인:832 ")
        self.checkRobotCnt += 1

        거래중 = False
        for t in self.매수거래시간:
            if t[0] <= current_str and current_str <= t[1]:
                거래중 = True
                # 한번만 실행하도록 함. by david
                if self.StartStock == True:
                    #todo: 테스트용 전체 삭제
                    # self.주문기록삭제('')
                    self.StartStock = False
                    self.XQ_t0424.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 단가구분='1', 체결구분='0', 단일가구분='0', 제비용포함여부='1', CTS_종목번호='')
                                                                                            # 체결구분-> 0 전체 잔고 조회
                    print(" 계좌조회요청...t0424\r\n")
        self.매수거래중 = 거래중

        거래중 = False
        for t in self.매도거래시간:
            if t[0] <= current_str and current_str <= t[1]:
                거래중 = True
        self.매도거래중 = 거래중

        #TODO: 특정시간의 강제매도
        #------------------------------------------------------------
        if self.일괄매도시각.strip() is not "":
            if self.일괄매도시각 < current_str and self.전량매도 == False:
                self.전량매도 = True

                if len(self.portfolio.keys()) > 0:
                    try:
                        print(" 특정시간 강제 매도 포트폴리오 개수 :", len(self.portfolio.keys()), self.portfolio.items())
                    except:
                        print("* 프린터 에러 발생 라인:865 ")

                    for k,v in self.portfolio.items():
                        단축코드 = v.종목코드
                        수량 = v.수량
                        종목명 = v.종목명
                        주문가 = '0'
                        호가유형코드 = '03'
                        self.매도Lock[단축코드] = 'S'
                        self.금일매도종목.append(단축코드)

                        if 수량 > 0:
                            try:
                                print(" 특정시간 강제 시장가 매도:", 종목명)
                            except:
                                print("* 프린터 에러 발생 라인:880 ")

                            # 매수 주문 후 매수 완료가 될때 까지 리얼데이터 임시 정지..
                            try:
                                if 단축코드 in self.kospi_codes:
                                    if type(self.XR_S3_) is not type(None):
                                        self.XR_S3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                        print(" S3 코스피 리얼데이터 종목 임시 정지..", 단축코드)
                                if 단축코드 in self.kosdaq_codes:
                                    if type(self.XR_K3_) is not type(None):
                                        self.XR_K3_.UnadviseRealDataWithKey(종목코드=단축코드)
                                        print(" K3 코스닥 리얼데이터 종목 임시 정지...", 단축코드)
                            except Exception as e:
                                print(f"리얼데이터 정지... 오류: {e}")

                            self.QA_CSPAT00600.Query(계좌번호=self.계좌번호, 입력비밀번호=self.비밀번호, 종목번호=단축코드, 주문수량=수량, 주문가=주문가, 매매구분=self.매도,
                                                     호가유형코드=호가유형코드, 신용거래코드=self.신용거래코드, 주문조건구분=self.조건없음)
                        else:
                            try:
                                print(" 특정시간 강제 매도 가능 수량이 0, 강제 매도할 종목이 아니다. 에러 발생함.", 종목명)
                            except:
                                print("* 프린터 에러 발생 라인:888 ")
                        # ToTelegram(__class__.__name__ + "일괄매도 : %s %s 주문수량:%s 주문가격:%s" % (단축코드, 종목명, 수량, 주문가))

                        현재시각 = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        매수가 = self.portfolio[단축코드].매수가
                        수량 = self.portfolio[단축코드].수량
                        매수일 = self.portfolio[단축코드].매수일.strftime("%Y-%m-%d %H:%M:%S")
                        data = [self.Name, self.UUID, 현재시각, 단축코드, 종목명, '일괄매도', 매수가, 주문가, 수량, 매수일, 0]
                        self.주문기록(data=data)

                else:
                    try:
                        print(" 강제 매도 포트폴리오가 개수가 0, 강제매도할 종목이 없음.")
                    except:
                        print("* 프린터 에러 발생 라인:902 ")

        #todo 먼저 체결 수량을 확인하고 그 수량을 일괄 취소 시킨다. 1초마다 주문번호 리스트를 요청한다.
        if self.일괄취소시각 < current_str and self.미체결용 == False:
        # if self.미체결용 == False:
            self.미체결용 = True
            print(" 미체결 종목 확인")
            self.XQ_t0425.Query(계좌번호=self.계좌번호, 비밀번호=self.비밀번호, 종목번호='', 체결구분='2', 매매구분='2', 정렬순서='1', 주문번호='')


    def Run(self, flag=True, parent=None):
        if self.running == flag:
            return

        self.parent = parent
        self.running = flag
        ret = 0
        if flag == True:
            ToTelegram("로직 [%s]을 시작합니다." % (__class__.__name__))

            self.QueryInit()

            self.clock = QtCore.QTimer()
            self.clock.timeout.connect(self.OnClockTick)
            self.clock.start(1000)
            self.전량매도 = False
            self.StartStock = True
            # self.검증플래그 = True

            self.금일매도종목 = []
            self.주문번호리스트 = []
            self.매수Lock = dict()
            self.매도Lock = dict()

            self.계좌번호, self.비밀번호 = parent.Account(구분='종합매매')
            print(self.계좌번호, self.비밀번호)

            with sqlite3.connect(self.DATABASE) as conn:
                query = 'select 단축코드,종목명,ETF구분,구분 from 종목코드'
                self.종목코드테이블 = pdsql.read_sql_query(query, con=conn)
                self.kospi_codes = self.종목코드테이블.query("구분=='1'")['단축코드'].values.tolist()
                self.kosdaq_codes = self.종목코드테이블.query("구분=='2'")['단축코드'].values.tolist()

            # todo 거래 설정 시간에 동작하도록 할도록 검토!
            # self.XQ_t1857 = t1857(parent=self, 식별자=uuid.uuid4().hex)
            # self.XQ_t1857.Query(실시간구분='0', 종목검색구분='F', 종목검색입력값=self.ACF파일)          # 실시간 구분 0으로 설정

            self.QA_CSPAT00600 = CSPAT00600(parent=self)
            self.QA_CSPAT00800 = CSPAT00800(parent=self)
            self.XQ_t0425 = t0425(parent=self)
            self.XQ_t0424 = t0424(parent=self)
            self.XQ_t1302 = t1302(parent=self)
            self.XQ_t8412 = t8412(parent=self)
            self.XR_S3_ = S3_(parent=self)
            self.XR_K3_ = K3_(parent=self)
            self.XQ_ChartIndex = ChartIndex(parent=self)

            # self.XR_SC0 = SC0(parent=self)
            self.XR_SC1 = SC1(parent=self)
            # self.XR_SC2 = SC2(parent=self)
            # self.XR_SC3 = SC3(parent=self)
            # self.XR_SC4 = SC4(parent=self)

            # self.XR_SC0.AdviseRealData()
            self.XR_SC1.AdviseRealData()
            # self.XR_SC2.AdviseRealData()
            # self.XR_SC3.AdviseRealData()
            # self.XR_SC4.AdviseRealData()

        else:
            if self.XQ_t1857 is not None:
                self.XQ_t1857.RemoveService()
                self.XQ_t1857 = None

            if self.clock is not None:
                try:
                    self.clock.stop()
                except Exception as e:
                    pass
                finally:
                    self.clock = None

            try:
                if self.XR_S3_ != None:
                    self.XR_S3_.UnadviseRealData()
            except Exception as e:
                pass
            finally:
                # self.XR_S3_ = None
                pass

            try:
                if self.XR_K3_ != None:
                    self.XR_K3_.UnadviseRealData()
            except Exception as e:
                pass
            finally:
                # self.XR_K3_ = None
                pass

            # if self.XR_SC0 != None:
            #     self.XR_SC0.UnadviseRealData()

            try:
                if self.XR_SC1 != None:
                    self.XR_SC1.UnadviseRealData()
            except Exception as e:
                pass
            finally:
                # self.XR_SC1 = None
                pass

            # if self.XR_SC2 != None:
            #     self.XR_SC2.UnadviseRealData()
            # if self.XR_SC3 != None:
            #     self.XR_SC3.UnadviseRealData()
            # if self.XR_SC4 != None:
            #     self.XR_SC4.UnadviseRealData()

            # self.QueryInit()

            # self.parent = None

def myReferenceRod(code,cnt, df, nowOpen, nowHigh, nowClose):
    # 기분봉평가 = None
    checkDaycnt = 0

    for i in range(2, cnt):
        oi_diff = df['고가기준등락율'].values[i] - df['저가기준등락율'].values[i]
        if oi_diff > 20.0 and df['등락율'].values[i] > 15.0 and df['회전율'].values[i] > 15:
            if df['시가'].values[i] < nowClose*0.95 and df['종가'].values[i] > nowClose*1.05:
                checkDaycnt = i
                print("  날찌: %s" % (df['날짜'].values[i]))
                break

    return checkDaycnt


def draw_candle_chart(sample_df, lines, region=None):
    # create figure
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figwidth(10)

    # define width of candlestick elements
    width = .4
    width2 = .05

    # define up and down prices
    up = sample_df[sample_df.종가 >= sample_df.시가]
    down = sample_df[sample_df.종가 < sample_df.시가]

    # define colors to use
    col1 = 'green'
    col2 = 'red'

    # plot up prices
    plt.bar(up.index, up.종가 - up.시가, width, bottom=up.시가, color=col1)
    plt.bar(up.index, up.고가 - up.종가, width2, bottom=up.종가, color=col1)
    plt.bar(up.index, up.저가 - up.시가, width2, bottom=up.시가, color=col1)

    # plot down prices
    plt.bar(down.index, down.종가 - down.시가, width, bottom=down.시가, color=col2)
    plt.bar(down.index, down.고가 - down.시가, width2, bottom=down.시가, color=col2)
    plt.bar(down.index, down.저가 - down.종가, width2, bottom=down.종가, color=col2)

    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    for x in lines:
        plt.hlines(x, xmin=sample_df.index[0], xmax=sample_df.index[-1])
        if region is not None:
            plt.fill_between(sample_df.index, x - x * region, x + x * region, alpha=0.4)

    # display candlestick chart
    plt.show()

def draw_line_chart(sample_df, lines, region=None, mavg=None):
    # create figure
    f = plt.figure()
    f.set_figwidth(15)
    f.set_figwidth(10)

    if mavg is not None:
        mavg_df = sample_df[["시가", "고가", "저가", "종가"]].rolling(window=mavg).mean()
        plt.plot(mavg_df.index, mavg_df.종가)
    else:
        plt.plot(sample_df.index, sample_df.종가)

    # rotate x-axis tick labels
    plt.xticks(rotation=45, ha='right')

    for x in lines:
        plt.hlines(x, xmin=sample_df.index[0], xmax=sample_df.index[-1])
        if region is not None:
            plt.fill_between(sample_df.index, x - x * region, x + x * region, alpha=0.4)

    # display candlestick chart
    plt.show()


def robot_loader():
    UUID = uuid.uuid4().hex
    robot = 실시간추적법(Name=ROBOT_NAME, UUID=UUID)
    return robot