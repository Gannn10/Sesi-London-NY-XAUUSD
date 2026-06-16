//+------------------------------------------------------------------+
//|                                                London_NY_Bot.mq5 |
//|                                             Putra Iqbal Amrullah |
//+------------------------------------------------------------------+
#property copyright "Putra Iqbal Amrullah"
#property link      ""
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

//--- inputs
input double   InpLotSize              = 0.01;        // Lot Size
input ulong    InpMagicNumber          = 999222;      // Magic Number
input ulong    InpDeviation            = 20;          // Deviation

// EMA
input int      InpEmaFast              = 50;          // EMA Fast Period
input int      InpEmaSlow              = 200;         // EMA Slow Period

// BOLLINGER BANDS
input int      InpBbPeriod             = 20;          // BB Period
input double   InpBbStdDev             = 2.0;         // BB Std Dev
input double   InpBbWidthMin           = 1.0;         // BB Width Min (%)
input double   InpToleransiBb          = 0.5;         // Toleransi BB

// RSI
input int      InpRsiPeriod            = 14;          // RSI Period
input double   InpRsiBullMin           = 52;          // RSI Bull Min (Breakout)
input double   InpRsiBearMax           = 48;          // RSI Bear Max (Breakout)
input double   InpRsiOverbought        = 60;          // RSI Overbought (Reversal)
input double   InpRsiOversold          = 40;          // RSI Oversold (Reversal)

// ASIA RANGE
input int      InpAsiaRangeStart       = 7;           // Asia Range Start Hour
input int      InpAsiaRangeEnd         = 15;          // Asia Range End Hour
input double   InpBreakoutBufferAtr    = 0.1;         // Breakout Buffer (ATR Multiplier)

// ATR
input int      InpAtrPeriod            = 14;          // ATR Period

// SESSION MULTIPLIER (from config_london_ny.py)
input double   InpLondonSlMultiplier   = 0.8;         // London SL Multiplier
input double   InpLondonTpMultiplier   = 1.0;         // London TP Multiplier
input double   InpNySlMultiplier       = 1.0;         // NY SL Multiplier
input double   InpNyTpMultiplier       = 1.2;         // NY TP Multiplier

// SESSION TIME (WIB / Broker Time matching)
input int      InpJamLondonBuka        = 15;          // London Open Hour
input int      InpJamLondonTutup       = 20;          // London Close Hour
input int      InpJamNyBuka            = 20;          // NY Open Hour
input int      InpJamNyTutup           = 24;          // NY Close Hour
input int      InpMaxTradePerSesi      = 999;         // Max Trade per Session

// RISK CONTROL
input int      InpMaxLossBeruntun      = 2;           // Max Loss Beruntun
input int      InpPauseSetelahLoss     = 60;          // Pause Setelah Loss (menit)

// Global variables
CTrade         trade;
CSymbolInfo    symInfo;
CPositionInfo  posInfo;

int            handle_ema_fast;
int            handle_ema_slow;
int            handle_bb;
int            handle_rsi;
int            handle_atr;

double         asia_high = 0.0;
double         asia_low = 0.0;
int            trade_count = 0;
datetime       last_trade_day = 0;
datetime       last_bar_time = 0;

int            loss_beruntun = 0;
datetime       pause_until = 0;
datetime       last_deal_time_checked = 0;

enum ENUM_SESSION_MODE {
   SESSION_INACTIVE = 0,
   SESSION_LONDON   = 1,
   SESSION_NEW_YORK = 2
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviation);
   trade.SetTypeFilling(ORDER_FILLING_IOC); 

   if(!symInfo.Name(_Symbol)) return(INIT_FAILED);
   
   handle_ema_fast = iMA(_Symbol, PERIOD_CURRENT, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   handle_ema_slow = iMA(_Symbol, PERIOD_CURRENT, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   handle_bb       = iBands(_Symbol, PERIOD_CURRENT, InpBbPeriod, 0, InpBbStdDev, PRICE_CLOSE);
   handle_rsi      = iRSI(_Symbol, PERIOD_CURRENT, InpRsiPeriod, PRICE_CLOSE);
   handle_atr      = iATR(_Symbol, PERIOD_CURRENT, InpAtrPeriod);
   
   if(handle_ema_fast == INVALID_HANDLE || handle_ema_slow == INVALID_HANDLE || 
      handle_bb == INVALID_HANDLE || handle_rsi == INVALID_HANDLE || handle_atr == INVALID_HANDLE) {
      Print("Gagal memuat indikator.");
      return(INIT_FAILED);
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   IndicatorRelease(handle_ema_fast);
   IndicatorRelease(handle_ema_slow);
   IndicatorRelease(handle_bb);
   IndicatorRelease(handle_rsi);
   IndicatorRelease(handle_atr);
}

//+------------------------------------------------------------------+
//| Get Session Mode                                                 |
//+------------------------------------------------------------------+
ENUM_SESSION_MODE GetSessionMode() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;
   
   if (hour >= InpJamLondonBuka && hour < InpJamLondonTutup) {
      return SESSION_LONDON;
   } else if (hour >= InpJamNyBuka && hour < InpJamNyTutup) {
      return SESSION_NEW_YORK;
   }
   return SESSION_INACTIVE;
}

//+------------------------------------------------------------------+
//| Calculate Asia Range                                             |
//+------------------------------------------------------------------+
void CalculateAsiaRange() {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   
   dt.hour = InpAsiaRangeStart;
   dt.min = 0;
   dt.sec = 0;
   datetime start_time = StructToTime(dt);
   
   dt.hour = InpAsiaRangeEnd;
   datetime end_time = StructToTime(dt);
   
   int start_bar = iBarShift(_Symbol, PERIOD_CURRENT, start_time);
   int end_bar = iBarShift(_Symbol, PERIOD_CURRENT, end_time);
   
   if (start_bar == -1 || end_bar == -1 || start_bar < end_bar) return;
   
   int count = start_bar - end_bar + 1;
   
   double high[], low[];
   if(CopyHigh(_Symbol, PERIOD_CURRENT, end_bar, count, high) > 0 && CopyLow(_Symbol, PERIOD_CURRENT, end_bar, count, low) > 0) {
      asia_high = high[ArrayMaximum(high)];
      asia_low = low[ArrayMinimum(low)];
      Print("Asia Range Terhitung. High: ", asia_high, " Low: ", asia_low);
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
   symInfo.RefreshRates();
   
   // Reset counter & asia range per hari
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   datetime current_day = dt.year * 10000 + dt.mon * 100 + dt.day;
   if (current_day != last_trade_day) {
       trade_count = 0;
       loss_beruntun = 0;
       pause_until = 0;
       last_trade_day = current_day;
       asia_high = 0.0;
       asia_low = 0.0;
   }
   
   if (TimeCurrent() < pause_until) return; // Sedang pause karena loss beruntun

   ENUM_SESSION_MODE mode = GetSessionMode();
   if(mode == SESSION_INACTIVE) return;
   
   if(trade_count >= InpMaxTradePerSesi) return;
   
   // Cek posisi aktif
   bool ada_posisi = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--) {
      if(posInfo.SelectByIndex(i)) {
         if(posInfo.Symbol() == _Symbol && posInfo.Magic() == InpMagicNumber) {
            ada_posisi = true;
            break;
         }
      }
   }
   
   // Cek deal yang baru ditutup untuk menghitung loss beruntun
   if(!ada_posisi) {
       if (HistorySelect(TimeCurrent() - 24*3600, TimeCurrent())) {
           int total_deals = HistoryDealsTotal();
           if (total_deals > 0) {
               ulong last_deal_ticket = HistoryDealGetTicket(total_deals - 1);
               datetime deal_time = (datetime)HistoryDealGetInteger(last_deal_ticket, DEAL_TIME);
               if(deal_time > last_deal_time_checked) {
                  last_deal_time_checked = deal_time;
                  if (HistoryDealGetInteger(last_deal_ticket, DEAL_MAGIC) == InpMagicNumber && 
                      HistoryDealGetInteger(last_deal_ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                      
                      double profit = HistoryDealGetDouble(last_deal_ticket, DEAL_PROFIT);
                      if (profit < 0) {
                          loss_beruntun++;
                          if (loss_beruntun >= InpMaxLossBeruntun) {
                              pause_until = TimeCurrent() + InpPauseSetelahLoss * 60;
                              loss_beruntun = 0;
                              Print("Max Loss Beruntun (", InpMaxLossBeruntun, "x) Tercapai! Pause bot selama ", InpPauseSetelahLoss, " menit.");
                              return;
                          }
                      } else {
                          loss_beruntun = 0;
                      }
                  }
               }
           }
       }
   }
   
   if(ada_posisi) return;
   
   // Kalkulasi Asia Range saat pertama kali masuk sesi London/NY di hari itu
   if(asia_high == 0.0 || asia_low == 0.0) {
      CalculateAsiaRange();
      if(asia_high == 0.0) return;
      if((asia_high - asia_low) < 1.0) { 
          Print("Range Asia terlalu sempit, skip trading hari ini.");
          trade_count = InpMaxTradePerSesi; 
          return;
      }
   }

   // Gunakan harga bid saat ini sebagai patokan penembusan
   double harga_close = symInfo.Bid(); 
   
   double ema_fast[], ema_slow[], bb_u[], bb_l[], bb_m[], rsi[], atr[];
   ArraySetAsSeries(ema_fast, true); ArraySetAsSeries(ema_slow, true);
   ArraySetAsSeries(bb_u, true); ArraySetAsSeries(bb_l, true); ArraySetAsSeries(bb_m, true);
   ArraySetAsSeries(rsi, true); ArraySetAsSeries(atr, true);
   
   // Copy data pada index 0 sesuai dengan implementasi python live tick df.iloc[-1]
   if(CopyBuffer(handle_ema_fast, 0, 0, 1, ema_fast) <= 0) return;
   if(CopyBuffer(handle_ema_slow, 0, 0, 1, ema_slow) <= 0) return;
   if(CopyBuffer(handle_bb, 1, 0, 1, bb_u) <= 0) return; 
   if(CopyBuffer(handle_bb, 2, 0, 1, bb_l) <= 0) return; 
   if(CopyBuffer(handle_bb, 0, 0, 1, bb_m) <= 0) return; 
   if(CopyBuffer(handle_rsi, 0, 0, 1, rsi) <= 0) return;
   if(CopyBuffer(handle_atr, 0, 0, 1, atr) <= 0) return;
   
   double e_fast = ema_fast[0];
   double e_slow = ema_slow[0];
   double upper_band = bb_u[0];
   double lower_band = bb_l[0];
   double mid_band = bb_m[0];
   double r = rsi[0];
   double a = atr[0];
   
   if(mid_band == 0.0) return;
   
   double bb_width = ((upper_band - lower_band) / mid_band) * 100.0;
   
   double sl_mult = 1.0, tp_mult = 1.0;
   if(mode == SESSION_LONDON) {
      sl_mult = InpLondonSlMultiplier;
      tp_mult = InpLondonTpMultiplier;
   } else if (mode == SESSION_NEW_YORK) {
      sl_mult = InpNySlMultiplier;
      tp_mult = InpNyTpMultiplier;
   }
   
   double buffer_atr = a * InpBreakoutBufferAtr;
   bool trend_up = (e_fast > e_slow && harga_close > e_fast);
   bool trend_down = (e_fast < e_slow && harga_close < e_fast);
   
   // LOGIKA BREAKOUT
   bool breakout_buy_break = harga_close > (asia_high + buffer_atr);
   bool breakout_buy_bb = harga_close > upper_band;
   bool breakout_buy_rsi = r > InpRsiBullMin;
   
   bool breakout_sell_break = harga_close < (asia_low - buffer_atr);
   bool breakout_sell_bb = harga_close < lower_band;
   bool breakout_sell_rsi = r < InpRsiBearMax;
   
   bool signal_breakout_buy = breakout_buy_break && breakout_buy_bb && breakout_buy_rsi && trend_up;
   bool signal_breakout_sell = breakout_sell_break && breakout_sell_bb && breakout_sell_rsi && trend_down;
   
   // LOGIKA REVERSAL
   bool reversal_buy_bb = harga_close <= lower_band;
   bool reversal_buy_rsi = r < InpRsiOversold;
   
   bool reversal_sell_bb = harga_close >= upper_band;
   bool reversal_sell_rsi = r > InpRsiOverbought;
   
   // Tambahkan filter tren agar tidak menangkap pisau jatuh
   bool signal_reversal_buy = reversal_buy_bb && reversal_buy_rsi && trend_up;
   bool signal_reversal_sell = reversal_sell_bb && reversal_sell_rsi && trend_down;
   
   bool valid_buy = signal_breakout_buy || signal_reversal_buy;
   bool valid_sell = signal_breakout_sell || signal_reversal_sell;
   
   if(valid_buy || valid_sell) {
      bool is_breakout = signal_breakout_buy || signal_breakout_sell;
      
      // Filter BB Width Min HANYA untuk Breakout
      if(is_breakout && bb_width <= InpBbWidthMin) {
         return; 
      }
      
      string strategy_name = is_breakout ? "Breakout" : "Reversal";
      string s_mode = (mode == SESSION_LONDON) ? "LONDON" : "NEW_YORK";
      
      if(valid_buy) {
         double ask = symInfo.Ask();
         double sl = ask - (a * sl_mult);
         double tp = ask + (a * tp_mult);
         
         if(trade.Buy(InpLotSize, _Symbol, ask, sl, tp, "BUY " + s_mode + " " + strategy_name)) {
            trade_count++;
            Print("ORDER BUY BERHASIL! TP: ", tp, " | SL: ", sl);
         } else {
            Print("ORDER BUY GAGAL: ", trade.ResultRetcodeDescription());
         }
      } 
      else if(valid_sell) {
         double bid = symInfo.Bid();
         double sl = bid + (a * sl_mult);
         double tp = bid - (a * tp_mult);
         
         if(trade.Sell(InpLotSize, _Symbol, bid, sl, tp, "SELL " + s_mode + " " + strategy_name)) {
            trade_count++;
            Print("ORDER SELL BERHASIL! TP: ", tp, " | SL: ", sl);
         } else {
            Print("ORDER SELL GAGAL: ", trade.ResultRetcodeDescription());
         }
      }
   }
}
//+------------------------------------------------------------------+
