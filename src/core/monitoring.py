"""
系统监控模块
提供系统健康状态监控、性能指标收集和异常检测功能
"""

import time
import psutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum

from .logging import get_logger, LoggerMixin
from .exceptions import TradingSystemException


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"      # 仪表
    HISTOGRAM = "histogram"  # 直方图
    TIMER = "timer"      # 计时器


@dataclass
class HealthCheck:
    """健康检查结果"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.status == HealthStatus.HEALTHY
    
    def is_critical(self) -> bool:
        """检查是否严重"""
        return self.status == HealthStatus.CRITICAL


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetatus=Healtst      
          alth',ess_henent='proc       compo
         eck(althChturn He  re         as e:
  xceptionxcept E
        e     
               )ds}
    m_threas': nuthreadnum_  details={'            进程健康",
  essage="          mY,
      .HEALTHStatuslthstatus=Hea               
 h',_healt='process  component            k(
  Checealthturn H        re    
          
            )     ds}
 threa: num_um_threads'details={'n                   reads}",
 : {num_th过多=f"线程数age       mess            ,
 WARNINGtatus.HealthSus=  stat                ,
  s_health't='proces componen           
        eck(hChrn Healt  retu             100:
  eads >hrnum_tf     i     eads()
   ess.num_thr= proceads     num_thr     # 检查线程数
                        

           )    "
    "进程处于僵尸状态  message=                  .CRITICAL,
lthStatusus=Hea  stat         
         _health','processt=    componen        
        k(ealthChecrn H       retu         US_ZOMBIE:
psutil.STATtatus() == f process.s          i 检查进程状态
          #  
            
  s()ocesl.Prtisu = p   process     try:
    
        ""进程健康"""检查  "  heck:
    althCf) -> Hes_health(selk_proceschec   def _ )
    
            }
      t
  censk_per': di_percentdisk    '            
nt,memory_percepercent': ry_  'memo      
        ent,: cpu_perccpu_percent'         '
          details={
         message,message=      
      tatus,  status=s          urces',
stem_resont='sympone      co
      lthCheck(ea Hreturn     
         "
  源正常"系统资 else esag) if messin(messages".jo = ";    message     
       f}%")
 rcent:.1: {disk_pef"磁盘使用率警告ges.append(   messa       .WARNING
  althStatus= He     status          :
  s.HEALTHYStatu== Healthus if stat     d:
       esholrning_thrself.disk_wa >= ercentsk_pif di   el   %")
  t:.1f}{disk_percen磁盘使用率严重: .append(f"sages  mes         ITICAL
 lthStatus.CR = Heatatus        s   
 shold:_threicalsk_crit self.diercent >= if disk_p       
        
%")percent:.1f}y_告: {memor使用率警end(f"内存ppsages.a    mes       .WARNING
 althStatus = He   status           LTHY:
  s.HEA HealthStatu==us      if stat     ld:
  ning_threshory_warmemont >= self.rcery_peelif memo    %")
    .1f}y_percent:mor重: {med(f"内存使用率严sages.appen  mes         
 ICALCRITtatus.ealthS = Hatus st    
       _threshold:iticaly_cr.memornt >= selfemory_perce     if m  
         1f}%")
:.ntcpu_perce"CPU使用率警告: {fges.append(  messa
          NGRNIthStatus.WAs = Heal     statu
       d:_thresholngnif.cpu_warsel >= ntu_percelif cp
        ef}%")ercent:.1_p使用率严重: {cpuCPUf"d(sages.appen  mes   
       us.CRITICAL= HealthStattus      sta    old:
   l_threshritica self.cpu_c_percent >=     if cpu       
   
  = []  messagesHY
      tus.HEALTStaealth H  status =
      # 判断状态  
             nt
 '/').percek_usage(til.dis = psucentdisk_per      t
  cen.permory()_meuall.virt = psutircent  memory_pe     val=0.1)
 internt(l.cpu_perce = psutipu_percent  c  ""
    系统资源""检查"      ":
  heckealthC> H) -sources(selfk_system_ref _chec de
    
   执行失败", e)健康检查 {name} ror(f"og_erlf.l  se             e:
  Exception aspt         exce
               }")
     ult.messagees{r {name} 警告: ning(f"健康检查f.log_warsel                    
    e:ls       e       
      ssage}")me严重: {result.name} "健康检查 {or(flf.log_err      se             ):
     al(s_critic if result.i                   thy():
is_healult.f not res    i          果
  # 记录健康检查结                 
         
      eck_func()result = ch                try:
      ():
      s.itemsheckalth_cc in self.heck_funme, cher na
        fo"""康检查"""执行健     
   :self)cks(th_cheorm_heal   def _perf e)
    
 失败",("收集系统指标og_errorlf.l         se as e:
   on Exceptipt    exce        
  ))
                 '
  unit='count              
 e.GAUGE,ricTyptype=Met   metric_            ,
 eads()num_thrs.rocesalue=p           vds',
     eacess.thrpro     name='           ric(
formanceMet(Perricmettor.record__collecmetricelf.        s    ss()
util.Proceess = ps   proc
         息       # 进程信    
             
      ))B'
         unit='M      ER,
       OUNTMetricType.C_type=      metric       024),
   (1024 * 1cv / bytes_realue=net_io.      v      ,
    rk.recv'.netwostemsye='nam               
 tric(ceMePerformanrd_metric(ector.recoc_coll.metri    self            ))
 
          unit='MB'            ,
 e.COUNTERetricTypype=M   metric_t     ),
        024 * 1024 (1 /s_senttet_io.byalue=ne         v  ,
     sent'k.em.networname='syst            ic(
    rmanceMetretric(Perfoor.record_mc_collectmetri    self.       ers()
 io_counttil.net_o = psu  net_i    网络IO
              #    
            ))
             t='%'
     uni      E,
     Type.GAUGMetricic_type=tr     me        
   ercent,k.pe=dis      valu    ',
      .percent.diske='system       nam      Metric(
   Performanceord_metric(ollector.rectric_cself.me     /')
       e('agdisk_ussk = psutil.        di用
     磁盘使        #
          )
            )MB'
            unit='    ,
      GEtricType.GAUtype=Me     metric_        4),
   1024 * 102d / (=memory.use   value        d',
     usemory.tem.me'sys    name=         Metric(
   (Performancecord_metric.rellectortric_co.me        self    ))
    '
        nit='% u           
    e.GAUGE,etricTypic_type=Metr        m
        rcent,ry.pealue=memo      v          ercent',
.p.memorytem  name='sys       ic(
       nceMetrrmaPerfoc(cord_metrire_collector.lf.metric se         
  emory()ual_m.virt = psutil  memory      用
        # 内存使         
                ))
%'
         unit='            E,
  AUGype.Gype=MetricT   metric_t            _percent,
 cpuue=       val        ent',
 rc.cpu.pe='systemame    n     
       tric(rmanceMetric(Perfoor.record_meic_collect   self.metr
         =1)intervalent(l.cpu_percent = psuti   cpu_perc
         使用率PU      # Ctry:
          ""
    "收集系统指标   """  ):
   etrics(self_system_m_collect
    def    al)
 heck_interv.cep(selfsle  time.         )
     "监控循环异常", error(lf.log_e   se            :
 ion as ecept Ex  except
          terval)elf.check_inp(sme.slee      ti
            # 等待下一次检查                      
        cks()
health_cheform__per   self.            康检查
    # 执行健                  
       )
    s(ystem_metricollect_s_c       self.       系统指标
  # 收集           
     try:           nning:
 elf.is_ru    while s
    监控循环""""""       
 loop(self):r_ _monito  def
    
  统监控已停止")info("系lf.log_se   =5)
     in(timeoutr_thread.jo self.monito   
        read:r_thtof self.moni   i
     alseg = Fs_runnin      self.i 
        return
           :
  runningt self.is_    if no   ""
 "停止监控"      ""lf):
   stop(se  
    def
  "系统监控已启动")f.log_info(     sel   
tart()ead.sonitor_thr     self.mTrue)
   mon=oop, daef._monitor_l(target=seleadThrhreading.r_thread = tnitolf.mo        see
ning = Tru.is_run   self
           
  urn         ret   在运行")
系统已"监控_warning(f.log   sel:
         _runningf.is      if sel""
  控""启动监   ""elf):
     t(s  def star
    
  ") {name}(f"注销健康检查:og_info     self.l       ks[name]
alth_chec.heel self         dhecks:
   elf.health_c sme in       if na"
 康检查"""""注销健
        name: str):k(self, h_checaltnregister_he u   
    def
 name}"): {健康检查nfo(f"注册self.log_i        nc
fuck_[name] = chehecksf.health_c   sel
     """健康检查  """注册  :
    hCheck])alt], Hellable[[k_func: Caecme: str, ch(self, nalth_checker_heaef regist
    d   
 ss_health)_check_proceth', self.rocess_heal'ph_check(healtgister_ self.rees)
       em_resourcsyst._check_ selfresources',stem_'sylth_check(egister_hea      self.r注册默认健康检查
       #   
       95.0
   shold =l_threisk_criticaf.del    s    
0.0 8eshold =g_thr.disk_warnin self     d = 90.0
  thresholy_critical_elf.memor
        s = 75.0sholdwarning_threelf.memory_0
        s= 90.threshold ritical__celf.cpu      s0.0
  reshold = 7rning_th self.cpu_wa        # 系统资源阈值
  
       one
      hread] = Nding.Tional[threaread: Optf.monitor_th     sel False
   ng =is_runni       self.= {}
 eck]] Che[[], Healthtr, Callabl: Dict[seckshealth_chlf.    setor()
    tricCollecMe = tortric_collec   self.me  erval
    = check_inteck_interval    self.ch()
    .__init__     super()= 60):
   int val: inter check_nit__(self,ef __i    d
    
器"""  """系统监控:
  xin)tor(LoggerMiMoniemyst
class Slear()

lf.metrics.c       se     
    e:   els   
      ar().cleme]f.metrics[na        sel      :
      metricsn self.f name i     i             if name:
      :
    th self.lock    wi"
    ""清除指标""      " None):
  ional[str] =me: Opt naetrics(self,_mclear
    def   }
       -1]
   st': values[  'late  
        s), len(valuees) /': sum(valu 'avg           es),
ax(valu    'max': m),
        values'min': min(      ),
      luesen(vaunt': l'co        n {
       retur]
     csmetrifiltered_ for m in = [m.values alue  v         
  turn {}
   re            s:
ered_metricot filt    if n      
    
  ff_time]mp >= cutotaimesics if m.ttrr m in me [m focs =etrid_mtere   filutes)
     _miniones=duratdelta(minutimee.now() - t= datetimoff_time    cut围内的指标
          # 过滤时间范     
   urn {}
   et r        
   ics:etrnot m if 
       rics(name).get_metcs = self       metri"""
 ""获取指标统计      "at]:
  str, flo> Dict[ = 60) -nutes: inturation_mitr, df, name: s(selmetric_statst_  def ge]
    
  s[name][-1elf.metric   return s        urn None
          ret0:
       ]) == me[natricslf.mecs or len(sef.metrinot in self name     ik:
        lf.loc  with se""
      标"新指获取最 """      tric]:
 rmanceMetional[Perfotr) -> Opme: s naelf,ic(satest_metret_ldef g  
    t:]
  )[-limis[name]elf.metricst(s  return li
           return []              
 metrics: in self.if name not         
   lf.lock:th se
        wi"""获取指标历史"""        etric]:
formanceMist[Per= 100) -> Lt r, limit: inname: st, trics(selfef get_me
    
    d(metric)ppend.ac.name]etrilf.metrics[m       se  
   tory)max_hislf.e(maxlen=seme] = dequmetric.nacs[tri.me      self        :
  csetri.melfin se not c.namif metri         .lock:
    self       with
 "记录指标""" "":
       ic)manceMetrc: Perfor metrielf,rd_metric(sef reco
    dk()
    ading.Locthreck = lf.lo       se{}
 r, deque] = : Dict[sttrics self.me   
    ax_historystory = mf.max_hi    sel   ):
  int = 1000istory:lf, max_ht__(seef __ini    
    d"""
标收集器  """指ctor:
  Colleass Metric

cl        }
format()
stamp.isof.timeamp': selimest't           _count,
 self.threadad_count':       'thrent,
      cess_cou self.proount':cess_cpro        '   cv_mb,
 work_reetlf.nb': se_recv_m 'network         ,
  _mbtwork_sentlf.ne': sent_mb_senetwork  '        t,
  cenperf.disk_cent': sel   'disk_per        mb,
 d_ry_usef.memoel s':ry_used_mb 'memo           ent,
_percf.memoryt': selcenemory_per'm            percent,
u_self.cpnt': pu_perce   'c         eturn {

        r转换为字典"""    """
    ny]: Astr,) -> Dict[elf(s to_dict
    def
    now)datetime.ctory=d(default_faiel f datetime =mp:timesta
    intead_count:    thrunt: int
 co   process_
 oatflk_recv_mb:   networoat
  t_mb: flk_senwor   netfloat
 t: rcen  disk_pe: float
  ory_used_mb    mement: float
ory_percmemat
    ent: floperc   cpu_"""
 ""系统指标"ics:
    etrystemMss
class Sla

@datac  }
it
      .unnit': self 'u       s,
    taggs': self.    'ta      ),
  rmat(estamp.isofotimmp': self.    'timesta
        ue,e.valic_typ: self.metrype'          'tvalue,
  ue': self.    'val       
 self.name,'name':       {
        return       """
为字典"转换     "":
   ct[str, Any]lf) -> Didict(se def to_  
   ""
  nit: str = 
    uct)_factory=difaultield(der] = f[str, sttags: Dict
    ow)ime.netdatry=efault_factold(dime = fie