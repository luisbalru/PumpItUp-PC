################################################
# RIPPER: PREPROCESAMIENTO Y CLASIFICACIÓN     #
# Autor: Luis Balderas Ruiz                    #
################################################
options(java.parameters = "-Xmx59g")
library(RWeka)
library(ggplot2)
library(rpart)
library(dplyr)


###############################################
# FUNCIONES PROPIAS

Accuracy = function(pred,etiq){
  return(length(pred[pred == etiq])/length(pred))
}

generaSubida = function(numero, test_id, prediccion){
  nombre = paste("submission_int",numero,".csv", sep="")
  submission = data.frame(test_id)
  submission$status_group = prediccion
  colnames(submission) = c("id", "status_group")
  write.csv(submission,file=nombre, row.names = FALSE)

}

# Lectura de datos
train = read.csv("training.csv")
labels = read.csv("training-labels.csv")
test = read.csv("test.csv")
train = merge(train, labels)

# PREPROCESAMIENTO

# Limpieza de datos

# Train --> construction_year

train$construction_year[train$construction_year == 0 & train$status_group == 'functional'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'functional']))
train$construction_year[train$construction_year == 0 & train$status_group == 'non functional'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'non functional']))
train$construction_year[train$construction_year == 0 & train$status_group == 'functional needs repair'] = round(mean(train$construction_year[train$construction_year != 0 & train$status_group == 'functional needs repair']))

test$construction_year = read.csv("cyt.csv")

# Creación de la variable estado en el test para que
# coincidan en número a la hora de hacer transformaciones
test$status_group = ""


#####################################################333
# VISUALIZACIÓN

# amount_tsh
ggplot(train, aes(x=longitude, y=latitude)) + geom_point(aes(colour=status_group))

train$longitude[train$region == "Arusha" & train$longitude == 0] =	36.55407
train$longitude[train$region=="Dar es Salaam" & train$longitude==0] = 39.21294
train$longitude[train$region=="Dodoma" & train$longitude==0] = 36.04196
train$longitude[train$region=="Iringa" & train$longitude==0] = 34.89592
train$longitude[train$region=="Kagera" & train$longitude==0] = 31.23309
train$longitude[train$region=="Kigoma" & train$longitude==0] = 30.21889
train$longitude[train$region=="Kilimanjaro" & train$longitude==0] = 37.50546
train$longitude[train$region=="Lindi" & train$longitude==0] = 38.98799
train$longitude[train$region=="Manyara" & train$longitude==0] = 35.92932
train$longitude[train$region=="Mara" & train$longitude==0] = 34.15698
train$longitude[train$region=="Mbeya" & train$longitude==0] = 33.53351
train$longitude[train$region=="Morogoro" & train$longitude==0] = 37.04678
train$longitude[train$region=="Mtwara" & train$longitude==0] = 39.38862
train$longitude[train$region=="Mwanza" & train$longitude==0] = 33.09477
train$longitude[train$region=="Pwani" & train$longitude==0] = 38.88372
train$longitude[train$region=="Rukwa" & train$longitude==0] = 31.29116
train$longitude[train$region=="Ruvuma" & train$longitude==0] = 35.72784
train$longitude[train$region=="Shinyanga" & train$longitude==0] = 33.24037
train$longitude[train$region=="Singida" & train$longitude==0] = 373950
train$longitude[train$region=="Tabora" & train$longitude==0] = 32.87830
train$longitude[train$region=="Tanga" & train$longitude==0] = 38.50195

ggplot(train, aes(x=longitude, y=latitude)) + geom_point(aes(colour=status_group))

data = rbind(train,test)

data$wpt_name = NULL
data$subvillage = NULL
data$ward = NULL
data$recorded_by = NULL
data$scheme_name = NULL
data$num_private = NULL
data$region_code = NULL
data$quantity_group = NULL
data$source_type = NULL
data$waterpoint_type_group = NULL
data$payment_type = NULL

data$funder = as.character(data$funder)
data$funder[data$funder == '' | data$funder== '0'] = 'Other'
data$funder[data$funder == 'Wfp/tnt' | data$funder == 'Wfp/tnt/usaid' | data$funder == 'Wfp/usaid' | data$funder == 'Wfp/usaid/tnt'] = 'Wfp/usaid'
data$funder[data$funder == 'Zao Water Spring' | data$funder == 'Zao Water Spring X'] = 'Zao Water Spring'
data$funder[data$funder == 'Yasi Naini' | data$funder == 'Yasini' | data$funder == 'Yasini Selemani'] = 'Yasini'
data$funder[data$funder == 'Zao' | data$funder == 'Zao Water Spring'] = 'Zao'
data$funder[data$funder == 'Wwf / Fores' | data$funder == 'Wwf'] = 'Wwf'
data$funder[data$funder == 'Wvt' | data$funder == 'Wvt Nakombo Adp'] = 'Wvt'
data$funder[data$funder == 'Wua' | data$funder == 'Wua And Ded' | data$funder == 'Wug And Ded'] = 'Wua'
data$funder[data$funder == 'Wrssp' | data$funder == 'Wsdo' | data$funder == 'Wsdp' | data$funder == 'Wsdp & Sdg' | data$funder == 'Wspd' | data$funder == 'Wsdo' | data$funder == 'Wssp' | data$funder == 'Wdsp' | data$funder == 'Wdp' | data$funder == 'Wfp' | data$funder == 'Wfp/usaid'] = 'Wsdp'
data$funder[data$funder == 'World Bank' | data$funder == 'World Bank/government' | data$funder == 'W0rld  Bank'] = 'World Bank'
data$funder[data$funder == 'World Vision' | data$funder == 'World Vision/ Kkkt' | data$funder == 'World Vision/adra' | data$funder == 'World Vision/rc Church' | data$funder == 'Worldvision'] = 'Worldvision'
data$funder[data$funder == 'Women Fo Partnership' | data$funder == 'Women For Partnership'] = 'Women For Partnership'
data$funder[data$funder == 'William Acleus' | data$funder == 'Williamson Diamond Ltd' | data$funder == 'Wilson'] = 'Wilson'
data$funder[data$funder == 'Watu Wa Marekani' | data$funder == 'Watu Wa Ujerumani'] = 'Watu Wa Marekani'
data$funder[data$funder == 'Water User As' | data$funder == 'Water User Associat' | data$funder == 'Water User Group'] = 'Water User As'
data$funder[data$funder == 'Water Aid/sema' | data$funder == 'Water Aid/dwe' | data$funder == 'Water Aid /sema' | data$funder == 'Water /sema' | data$funder == 'Wate Aid/sema' | data$funder == 'Water'] = 'Water'
data$funder[data$funder == 'Wanan' | data$funder == 'Wananchi'] = 'Wananchi'
data$funder[data$funder == 'Wamissionari Wa Kikatoriki' | data$funder == 'Wamisionari Wa Kikatoriki'] = 'Wamisionari Wa Kikatoriki'
data$funder[data$funder == 'W.D & I.' | data$funder == 'W.D.&.I.' | data$funder == 'W.D &' | data$funder == 'W.F.D.P' | data$funder == 'W.C.S' | data$funder == 'W.B' | data$funder == 'W'] = 'W.F.D.P'
data$funder[data$funder == 'Vwc' | data$funder == 'Vwcvc' | data$funder == 'Vwcvwc' | data$funder == 'Vwt' | data$funder == 'Vw' | data$funder == 'Vn'] = 'Vw'
data$funder[data$funder == 'Villa' | data$funder == 'Villaers' | data$funder == 'Village' | data$funder == 'Village Authority' | data$funder == 'Village Committee' | data$funder == 'Village Communi' | data$funder == 'Village Community' | data$funder == 'Village Contributio' | data$funder == 'Village Council' | data$funder == 'Village Council/ Haydom Luther' | data$funder == 'Village Council/ Rose Kawala' | data$funder == 'Village Fou' | data$funder == 'Village Fund' | data$funder == 'Village Gover' | data$funder == 'Village Government' | data$funder == 'Village Govt' | data$funder == 'Village Office' | data$funder == 'Village Res' | data$funder == 'Village Water Commission' | data$funder == 'Villager' | data$funder == 'Villagers' | data$funder == 'Villagers Mpi' | data$funder == 'Villages' | data$funder == 'Villege Council' | data$funder == 'Villegers' | data$funder == 'Villlage Contributi' | data$funder == 'Vn'] = 'Village'
data$funder[data$funder == 'Vickfis' | data$funder == 'Vicfish Ltd' | data$funder == 'Vicfish' | data$funder == 'Vi'] = 'Vicfish'
data$funder[data$funder == 'Vifafi' | data$funder == 'Vififi']  = 'Vifafi'
data$funder[data$funder == 'Water Authority' | data$funder == 'Water Board' | data$funder == 'Water Department' | data$funder =='Water Of Water' | data$funder == 'Water Project Mbawala Chini' | data$funder == 'Water Se'  | data$funder == 'Water Sector Development' | data$funder == 'Water User As' | data$funder == 'Wateraid'] = 'Water'
data$funder[data$funder == 'Usaid' | data$funder =='Usaid/wpf'] = 'Usaid'
data$funder[data$funder == 'Usa Embassy' | data$funder == 'Us Embassy' | data$funder == 'U.S.A'] = 'Usa Embassy'
data$funder[data$funder == 'Unice' | data$funder == 'Unice/ Cspd' | data$funder == 'Unicef' | data$funder == 'Unicef/ Csp' | data$funder == 'Unicef/african Muslim Agency' | data$funder == 'Unicef/central' | data$funder == 'Unicef/cspd' | data$funder == 'Uniceffinida German Tanzani' | data$funder == 'Uniceffinidagermantanzania' | data$funder == 'Uniceg' | data$funder == 'Unicet' | data$funder == 'Unicrf' | data$funder == 'Unise' | data$funder == 'Uniseg'] = 'Unicef'
data$funder[data$funder == 'Unhcr' | data$funder == 'Unhcr/danida' | data$funder == 'Unhcr/government']  = 'Unhcr'
data$funder[data$funder == 'Undp' | data$funder == 'Undp/aict' | data$funder == 'Undp/ilo' | data$funder == 'Unp/aict'] = 'Undp'
data$funder[data$funder == 'Un' | data$funder == 'Un Habitat' | data$funder == 'Un/wfp'] = 'Un'
data$funder[data$funder == 'Usaid' | data$funder == 'Usaid/wfp'] = 'Usaid'
data$funder[data$funder == 'Ubalozi Wa Japani' | data$funder == 'Ubalozi Wa Marekani' | data$funder == 'Ubalozi Wa Marekani/dwe'] = 'Ubalozi Wa Marekani/dwe'
data$funder[data$funder == 'Tz As' | data$funder == 'Tz Japan' | data$funder == 'Tz/japan Embass'] = 'Tz Japan'
data$funder[data$funder == 'Umoja' | data$funder == 'Umoja Makanisa Pentekoste Tz'] = 'Umoja'
data$funder[data$funder == 'Total Land Care' | data$funder == 'Total Landcare' | data$funder == 'Totaland Care' | data$funder == 'Totoland' | data$funder == 'Totoland Care'] = 'Total Landcare'
data$funder[data$funder == 'Tom' | data$funder == 'Tomas Kasmil'] = 'Tom'
data$funder[data$funder == 'Tkc' | data$funder == 'Tlc' | data$funder == 'Tlc/community' | data$funder == 'Tlc/emmanuel Kasoga' |data$funder == 'Trc' | data$funder == 'Tlc/jenus Malecha' | data$funder == 'Tlc/john Majala' | data$funder == 'Tlc/nyengesa Masanja' | data$funder == 'Tlc/samora' | data$funder == "Tlc/seleman Mang'ombe" | data$funder == 'Tlc/sorri' | data$funder == 'Tlc/thimotheo Masunga' | data$funder == 'Tltc'] = 'Tlc'
data$funder[data$funder == 'Theo' | data$funder == 'Theonas Mnyama'] = 'Theo'
data$funder[data$funder == 'The Isla' | data$funder == 'The Islamic'] = 'The Islamic'
data$funder[data$funder == 'The Desk And Chair Foundat' | data$funder == 'The Desk And Chair Foundati'] = 'The Desk And Chair Foundati'
data$funder[data$funder == 'Tasaf' |data$funder == 'Taasaf' | data$funder == 'Tasaf 1' | data$funder == 'Tasae' | data$funder == 'Tasa' | data$funder == 'Tasad' | data$funder == 'Tasaf And Lga' | data$funder == 'Tasaf And Mmem' | data$funder == 'Tasaf Ii' | data$funder == 'Tasaf/dmdd' | data$funder == 'Tasaf/tlc' | data$funder == 'Tasaf/village Community' | data$funder == 'Tasafu' | data$funder == 'Tasef' | data$funder == 'Tasf' | data$funder == 'Tasmin' | data$funder == 'Tassaf' | data$funder == 'Tassaf I' | data$funder == 'Tassaf Ii' | data$funder == 'Tassaf/ Danida'] = 'Tasaf'
data$funder[data$funder == 'Tgrs' | data$funder == 'Tgt' | data$funder == 'Tgts' | data$funder == 'Tgz'] = 'Tgz'
data$funder[data$funder == 'Tcrs' | data$funder == 'Tcrs/care' | data$funder =='Tcrs /government' | data$funder == 'Tcrs Kibondo' | data$funder == 'Tcrs.Tlc' | data$funder == 'Tcrs/care' | data$funder == 'Tcrs/village Community' | data$funder == 'Tcrst' | data$funder == 'Tdft' | data$funder == 'Tdrs'] = 'Tdrs'
data$funder[data$funder == 'Tansi' | data$funder == 'Tanza' | data$funder == 'Tanzakesho' | data$funder == 'Tanz Egypt Technical Cooper'| data$funder == 'Tanz Egypt Technical Coope' | data$funder == 'Tanz/egypt Technical  Co-op' | data$funder == 'Tanzaling' | data$funder == 'Tanzania' | data$funder == 'Tanzania /egypt' | data$funder == 'Tanzania And Egypt Cooperat' | data$funder == 'Tanzania Christian Service' | data$funder == 'Tanzania Compasion' | data$funder == 'Tanzania Egypt Technical Co Op' | data$funder == 'Tanzania Journey' | data$funder == 'Tanzania Na Egypt' | data$funder == 'Tanzania Nea Egypt' | data$funder == 'Tanzania/australia'] ='Tanzania'
data$funder[data$funder == 'Tag' | data$funder == 'Tag Church']  = 'Tag'
data$funder[data$funder == 'Tadeo' | data$funder == 'Tado' | data$funder == 'Ta' | data$funder == 'T'] = 'Tado'
data$funder[data$funder == 'Swiss Tr' | data$funder == 'Swiss If' | data$funder == 'Swisland/mount Meru Flowers' | data$funder == 'Swisland/ Mount Meru Flowers' | data$funder == 'Swifti'] = 'Swiss'
data$funder[data$funder == 'Swidish' | data$funder == 'Sweeden' | data$funder == 'Swedish Tandala Project' | data$funder == 'Swedish' | data$funder == 'Swidish'] = 'Sweden'
data$funder[data$funder == 'Solidame' | data$funder == 'Solidarm' | data$funder == 'Soliderm'] = 'Soliderm'
data$funder[data$funder == 'Snv' | data$funder == 'Snv Ltd' | data$funder == 'Snv-swash'] = 'Snv'
data$funder[data$funder == 'Sister Francis' | data$funder == 'Siter Fransis'] = 'Sister Francis'
data$funder[data$funder == 'Si' | data$funder == 'Sida'] = 'Si'
data$funder[data$funder == 'Shule' | data$funder == 'Shule Ya Msingi' | data$funder == 'Shule Ya Msingi Ufala' | data$funder == 'Shule Ya Sekondari Ipuli'] = 'Shule'
data$funder[data$funder == 'Serikali' | data$funder == 'Serikali Ya Kijiji'] = 'Serikali'
data$funder[data$funder == 'Sema' | data$funder == 'Sema S' | data$funder == 'Semaki' | data$funder == 'Semaki K'] = 'Sema'
data$funder[data$funder == 'Secondary Schoo' | data$funder == 'Secondary'] = 'Secondary'
data$funder[data$funder == 'Sda' | data$funder == 'Sda Church' | data$funder == 'Sdg' | data$funder == 'Sdp'] = 'Sda'
data$funder[data$funder == 'Songea District Council' | data$funder == 'Songea Municipal Counci'] = 'Songea Municipal Counci'
data$funder[data$funder == 'Scholastica Pankrasi' | data$funder == 'Schoo' | data$funder == 'School' | data$funder == 'School Adm9nstrarion' | data$funder == 'School Administration' |data$funder == 'School Capital'] = 'School'
data$funder[data$funder == 'Serikali' | data$funder == 'Serikari' |data$funder == 'Serikaru'] = 'Serikari'
data$funder[data$funder == 'Rv' | data$funder == 'Rvemp' | data$funder == 'Rw Ssp' | data$funder == 'Rwi' | data$funder=='Rwsp' | data$funder == 'Rwsso' | data$funder == 'Rwssp' | data$funder == 'Rwssp Shinyanga' | data$funder == 'Rwssp/wsdp' | data$funder == 'Rwsssp'] = 'Rwsssp'
data$funder[data$funder == 'Rotary' | data$funder == 'Rotary Club' | data$funder == 'Rotary Club Australia' | data$funder == 'Rotary Club Kitchener' | data$funder == 'Rotary Club Of Chico And Moshi' | data$funder == 'Rotary Club Of Usa And Moshi' | data$funder == 'Rotary I' | data$funder == 'Rotaty Club' | data$funder == 'Rotery C' | data$funder == 'Rotte'] = 'Rotary'
data$funder[data$funder == 'Roman' | data$funder == 'Romam Catholc/vil' | data$funder == 'Romam Catholic' | data$funder == 'Roman Ca' | data$funder == 'Roman Catholic' | data$funder == 'Roman Catholic Rulenge Diocese' | data$funder == 'Roman Cathoric' | data$funder == 'Roman Cathoric -kilomeni' | data$funder == 'Roman Cathoric Church' | data$funder == 'Roman Cathoric Same' | data$funder == 'Roman Cathoric-same' | data$funder == 'Roman Church'] = 'Roman Church'
data$funder[data$funder == 'Rudep' | data$funder == 'Rudep /dwe' | data$funder == 'Rudep/dwe'] = 'Rudep'
data$funder[data$funder == 'Rural' | data$funder == 'Rural Drinking Water Supply' | data$funder == 'Rural Water Department' | data$funder == 'Rural Water Supply' | data$funder == 'Rural Water Supply And Sanita' | data$funder == 'Rural Water Supply And Sanitat' ] = 'Rural'
data$funder[data$funder == 'Rc' | data$funder == 'Rc Cathoric' | data$funder == 'Rc Ch' | data$funder == 'Rc Churc' | data$funder == 'Re' | data$funder == 'Red Cross' | data$funder == 'Rdc' | data$funder == 'Rdws'|data$funder == 'Rc Church'| data$funder == 'Rc Church/centr' | data$funder == 'Rc Mi' | data$funder == 'Rc Missi' | data$funder == 'Rc Mission' | data$funder == 'Rc Missionary' | data$funder == 'Rc Mofu' | data$funder == 'Rc Msufi' | data$funder == 'Rc Njoro' | data$funder == 'Rc/dwe' | data$funder == 'Rc/mission' | data$funder == 'Rcchurch' | data$funder == 'Rcchurch/cefa'] = 'Rcchurch'
data$funder[data$funder == 'Quick' | data$funder == 'Quick Win' | data$funder == 'Quick Win Project' | data$funder == 'Quick Win Project /council' | data$funder == 'Quick Win/halmashauri' | data$funder == 'Quick Wings' | data$funder == 'Quick Wins' | data$funder == 'Quick Wins Scheme' | data$funder == 'Quicklw' | data$funder == 'Quickwi' | data$funder == 'Quickwins' | data$funder == 'Quik' | data$funder == 'Qwckwin' | data$funder == 'Qwekwin' | data$funder == 'Qwick Win' | data$funder == 'Qwickwin' | data$funder == 'Quwkwin' |data$funder == 'Qwiqwi'] = 'Quick'
data$funder[data$funder == 'Priva' | data$funder == 'Private' | data$funder == 'Private Co' | data$funder == 'Private Company' | data$funder == 'Private Individual' |data$funder == 'Private Individul' | data$funder == 'Private Institutions' | data$funder == 'Private Manager' | data$funder == 'Private Owned' | data$funder == 'Private Person'] = 'Private'
data$funder[data$funder == 'Ox' | data$funder == 'Oxfam' | data$funder == 'Oxfam Gb' | data$funder == 'Oxfarm' | data$funder == 'Oxfarm Gb'] = 'Oxfam'
data$funder[data$funder == 'Oikos' | data$funder == 'Oikos E .Africa/european Union' | data$funder == 'Oikos E.Africa/ European Union' | data$funder == 'Oikos E.Africa/european Union' | data$funder == 'Oikos E.Afrika'] = 'Oikos E.Afrika'
data$funder[data$funder == 'Norad' | data$funder == 'Norad /government' | data$funder == 'Norad/ Kidep' | data$funder == 'Norad/ Tassaf' | data$funder == 'Norad/ Tassaf Ii' | data$funder == 'Norad/government' | data$funder == 'Norad/japan' | data$funder == 'Norad/rudep'] = 'Norad'
data$funder[data$funder == 'Nerthlands' | data$funder == 'Nethalan' | data$funder == 'Natherland' | data$funder == 'Nethe' | data$funder == 'Netherla' | data$funder == 'Netherland' | data$funder == 'Netherlands'] = 'Netherlands'
data$funder[data$funder == 'Luthe' | data$funder == 'Lutheran' | data$funder == 'Lutheran Church'] = 'Lutheran'
data$funder[data$funder == 'Lg' | data$funder == 'Lga' | data$funder == 'Lga And Adb' | data$funder == 'Ldcgd'] = 'Lga'
data$funder[data$funder == 'Kkkt' | data$funder == 'Kkkt Canal' | data$funder == 'Kkkt Church' | data$funder == 'Kkkt Church S' | data$funder == 'Kkkt Dme' | data$funder == 'Kkkt Imbaseny' | data$funder == 'Kkkt Kolila' | data$funder == 'Kkt Leguruki' | data$funder == 'Kkkt Mareu' | data$funder == 'Kkkt Mso' | data$funder == 'Kkkt Ndrumangeni' | data$funder == 'Kkkt Usa' | data$funder == 'Kkkt_makwale' | data$funder == 'Kkkt-dioces Ya Pare'] = 'Kkkt'
data$funder[data$funder == 'Fathe' | data$funder == 'Father Bonifasi' | data$funder == 'Father W'] = 'Father'
data$funder[data$funder == 'Eu' | data$funder == 'European Union'] = 'Eu'
data$funder[data$funder == 'Dv' | data$funder == 'Dw' | data$funder == 'Dwarf' | data$funder == 'Dwe' | data$funder == 'Dwe And Veo' | data$funder == 'Dwe/anglican Church' | data$funder == 'Dwe/bamboo Projec' | data$funder == 'Dwe/norad' | data$funder == 'Dwe/rudep' | data$funder == 'Dwe/ubalozi Wa Marekani'] = 'Dwe'
data$funder[data$funder== 'Dwspd' | data$funder == 'Dwsp' | data$funder == 'Dwsp & Central Government' | data$funder == 'Dwspd' | data$funder == 'Dwssp' | data$funder == 'Dwst' | data$funder == 'Dwt'] = 'Dwt'
data$funder[data$funder == 'Ces (gmbh)' | data$funder == 'Ces(gmbh)'] = 'Ces'
data$funder = as.factor(data$funder)


data$funder = as.character(data$funder)
tabla_funder = table(data$funder)
minoritarios = names(tabla_funder[tabla_funder<=15])
data$funder[data$funder %in% minoritarios] = "otros"
data$funder = as.factor(data$funder)


data$installer = as.character(data$installer)
data$installer[data$installer == ''] = 0
data$installer[data$installer == '' |data$installer == '-' | data$installer== '0'] = 'Other'
data$installer[data$installer == 'Action Aid' | data$installer == 'ACTION AID' ] = 'Action Aid'
data$installer[data$installer == 'Cons' | data$installer == 'CONS' | data$installer == 'Consultant'] = 'Consultant'
data$installer[data$installer == 'Action Contre la Faim' | data$installer == 'Action Contre La Faim'] = 'Action Contre La Faim'
data$installer[data$installer == 'ACRA' | data$installer == 'Accra'] = 'Acra'
data$installer[data$installer == 'AAR' | data$installer == 'Aartisa'] = 'Aar'
data$installer[data$installer == 'Active KMK' | data$installer == 'Active MKM'] = 'Active MKM'
data$installer[data$installer == 'Af' | data$installer == 'AF'] = 'Af'
data$installer[data$installer == 'Luthe' | data$installer == 'Lutheran' | data$installer == 'lutheran church' | data$installer == 'Lutheran Church'] = 'Lutheran'
data$installer[data$installer == 'MACK DONALD CO LTD' | data$installer == 'MACK DONALD CONTRACTOR' | data$installer == 'MACK DONALD CONTRSCTOR' |data$installer == 'Mackd'] = 'Mack'
data$installer[data$installer == 'Ma' | data$installer == 'MA'] = 'Ma'
data$installer[data$installer == 'local' | data$installer == 'Local' | data$installer == 'local  technician' | data$installer == 'Local  technician' | data$installer == 'Local l technician' | data$installer == 'Local te' | data$installer == 'Local technical' | data$installer == 'local technical tec' | data$installer == 'Local technical tec' | data$installer == 'local technician' | data$installer =='Local technician' | data$installer == 'local technitian' | data$installer == 'Local technitian' | data$installer == 'Locall technician' | data$installer == 'Localtechnician'] = 'Local te'
data$installer[data$installer == 'LIVI' | data$installer == 'Livi'] = 'Livi'
data$installer[data$installer == 'Lion\'s club' | data$installer == 'lion\'s club'| data$installer == 'Lion\'s'| data$installer == 'LION\'S' | data$ installer == 'Lions club kilimanjaro'] = 'Lions'
data$installer[data$installer == 'LINDALA' | data$installer == 'Linda' | data$installer == 'Li'] = 'Li'
data$installer[data$installer == 'LGA' | data$installer == 'Lga'] = 'Lga'
data$installer[data$installer == 'Losa-kia water suppl' | data$installer == 'Losaa-Kia water supp' | data$installer == 'Losakia water supply'] = 'Losakia'
data$installer[data$installer == 'LWI' | data$installer == 'LWI &CENTRAL GOVERNMENT'] = 'LWI'
data$installer[data$installer == 'maendeleo ya jamii' | data$installer == 'Maendeleo ya jamii' | data$installer == 'Maendeleo'] = 'Maendeleo'
data$installer[data$installer == 'Magadini Makiwaru wa' | data$installer == 'Magadini-Makiwaru' | data$installer == 'Magadini-Makiwaru wa' | data$installer == 'Magani'] = 'Magadini'
data$installer[data$installer == 'M and P' | data$installer == 'Ma' | data$installer == 'M'] = 'M'
data$installer[data$installer == 'LVA Ltd' | data$installer == 'LVIA'] = 'LVA'
data$installer[data$installer == 'LOLMOLOKI' | data$installer == 'LOMOLOKI'] = 'Lomoloki'
data$installer[data$installer == 'Lawate fuka water su' | data$installer == 'lawatefuka water sup'] = 'Lawate'
data$installer[data$installer == 'Maji Tech' | data$installer == 'MAJI TECH' | data$installer == 'Maji tech Construction'] = 'Maji'
data$installer[data$installer == 'maji mugumu' | data$installer == 'MAJ MUGUMU' | data$installer == 'maji mugumu' | data$installer == 'MAJI MUGUMU'] = 'Maji mugumu'
data$installer[data$installer == 'Makonde' | data$installer == 'Makonde water population' | data$installer == 'Makonde water Population' | data$installer == 'Makonde water supply'] = 'Makonde'
data$installer[data$installer == 'Kuwait' | data$installer == 'KUWAIT' | data$installer == 'Kuwaiti' | data$instaler == 'kuwait'] = 'Kuwait'
data$installer[data$installer == 'KILI WATER' | data$installer == 'KILL WATER' | data$installer == 'Killflora /Community'  | data$installer == 'Kiliflora' | data$installer == 'Killflora/ Community' | data$installer == 'Killflora /Comunity'] = 'Kili'
data$installer[data$installer == 'Kiliwater' | data$installer == 'Kiliwater r'] = 'Kiliwater'
data$installer[data$installer == 'KK' | data$installer == 'Kkkt' | data$installer == 'KkKT' | data$installer == 'KKKT'] = 'Kkkt'
data$installer[data$installer == 'Ko' | data$installer == 'KOBERG' | data$installer == 'KOBERG Contractor'] = 'Koberg'
data$installer[data$installer == 'Adra' | data$installer == 'ADRA' | data$installer == 'Adra /Community' | data$installer == 'ADRA/Government' | data$installer == 'Adra/ Community' | data$installer == 'Adra/Community' | data$installer == 'ADRA/Government' | data$installer == 'Adrs'] = 'Adra'
data$installer[data$installer == 'DW#' | data$installer == 'DW$' | data$installer == 'DW' | data$installer == 'DW E' | data$installer == 'Dwe' |data$installer == 'DWE' | data$installer == 'DWE & LWI' | data$installer == 'DWE /TASSAF' | data$installer == 'DWE&' | data$installer == 'DWE/' | data$installer == 'DWE/Angelican church' | data$installer == 'DWE/TASSAF' | data$installer == 'DWE/Ubalozi wa Marekani' | data$installer == 'DWE}' | data$installer == 'DWEB' | data$installer == 'DWR'] = 'DWE'
data$installer[data$installer == 'D' | data$installer == 'D$L' | data$installer == 'Da' | data$installer == 'DA'] = 'DA'
data$installer[data$installer == 'DW#' | data$installer == 'DW$' | data$installer == 'DW' | data$installer == 'DW E' | data$installer == 'Dwe' |data$installer == 'DWE' | data$installer == 'DWE & LWI' | data$installer == 'DWE /TASSAF' | data$installer == 'DWE&' | data$installer == 'DWE/' | data$installer == 'DWE/Angelican church' | data$installer == 'DWE/TASSAF' | data$installer == 'DWE/Ubalozi wa Marekani' | data$installer == 'DWE}' | data$installer == 'DWEB' | data$installer == 'DWR'] = 'DWE'
data$installer[data$installer == 'Da' | data$installer == 'DA' | data$installer == 'DADIS' | data$installer == 'DADP' | data$installer == 'DADS' |data$installer == 'DADS/village community' | data$installer == 'DADS/village Community' | data$installer == 'DADS/Village community' ] ='DADS'
data$installer[data$installer == 'Concern' | data$installer == 'CONCERN' | data$installer == 'Concern /government' | data$installer == 'Concern/Government' | data$installer == 'Concen' | data$installer == 'CONCE' | data$installer == 'Conce'] ='Concern'
data$installer[data$installer == 'hesaw' | data$installer == 'HESAW' | data$installer == 'HESAWA' | data$installer == 'hesawa' | data$installer == 'Hesawa' |data$installer == 'HesaWa' | data$installer == 'HESAWQ' | data$installer == 'HESAWS' | data$installer == 'Hesawz' | data$installer == 'HESAWZ' | data$installer == 'Hesewa'] = 'Hesawa'
data$installer[data$installer == 'Masjid' | data$installer == 'Masjid Nnre' | data$installer == 'MasjId Takuar'] = 'Masjid'
data$installer[data$installer == 'Mbozi Hospital' | data$installer == 'Mbozi District Council' | data$installer == 'Mbozi Secondary School'] = 'Mbozi'
data$installer[data$installer == 'Maswi' | data$installer == 'MASWI' | data$installer == 'MASWI CO' | data$installer == 'Maswi company' | data$installer == 'Maswi Company' | data$installer == 'MASWI COMPANY' | data$installer == 'MASWI DRILL' | data$installer == 'MASWI DRILLING' | data$installer == 'Maswi drilling co ltd' | data$installer == 'MASWI DRILLING CO. LTD'] = 'Maswi'
data$installer[data$installer == 'MD' | data$installer == 'Mdala Contractor' | data$installer == 'MDALA Contractor' ]= 'Mdala'
data$installer[data$installer == 'Mara inter product' | data$installer == 'MARAFIN' | data$installer == 'marafip' | data$installer == 'Marafip' | data$installer == 'MARAFIP' ] = 'Mara'
data$installer[data$installer == 'KARUMBA BIULDIN' | data$installer == 'KARUMBA BIULDING COMPANY LTD' | data$installer == 'KARUMBA BIULDING CONTRACTOR' | data$installer == 'KARUMBA BUILDING COMPANY LTD'] = 'Karumba'
data$installer[data$installer == 'Jaica' | data$installer == 'JAICA' | data$installer == 'JAICA CO' | data$installer == 'JALCA'] = 'Jaica'
data$installer[data$installer == 'kuwait' | data$installer == 'Kuwait' | data$installer == 'Kuweit' | data$installer == 'kw'] = 'Kuwait'
data$installer[data$installer == 'JI' | data$installer == 'Jica' | data$installer == 'JICA' | data$installer == 'Jicks' | data$installer == 'Jika' | data$installer == 'JIKA' | data$installer == ' Jiks'] = 'Jica'
data$installer[data$installer == 'konoike' | data$installer == 'KONOIKE' ]= 'Konoike'
data$installer[data$installer == 'IDARA' | data$installer == 'Idara ya maji' | data$installer == 'Idara ya Maji'] = 'Idara'
data$installer[data$installer == 'Milenia' | data$installer == 'Mileniam' | data$installer == 'Mileniam projet' |data$installer == 'Mileniam project' | data$installer == 'Milenium'] = 'Milenia'
data$installer[data$installer == 'MBIUSA' | data$installer == 'MBIUWASA'] = 'MBIUSA'
data$installer[data$installer == 'Mark' | data$installer == 'Marke'] = 'Mark'
data$installer[data$installer == 'JANDU' | data$installer == 'JANDU PLUMBER  CO' | data$installer == 'JANDU PLUMBER CO'] = 'Jandu'
data$installer[data$installer == 'Halmashauri'| data$installer == 'Halmashauli' | data$installer == 'Halmashauri' | data$installer == 'HAlmashauli' | data$installer == 'Halimashauli'] = 'Halmashauri'
data$installer[data$installer == 'Ministry of water' | data$installer == 'MINISTRY OF WATER' | data$installer == 'Ministry of water engineer' | data$installer == 'MINISTRYOF WATER'] = 'Ministry of water'
data$installer[data$installer == 'Mi' | data$installer == 'MI'] = 'Mi'
data$installer[data$installer == 'Mh Kapuya' | data$installer == 'MH Kapuya'] = 'Mh Kapuya'
data$installer[data$installer == 'Missio' | data$installer == 'Missi'] = 'Missi'
data$installer[data$installer == 'Japan' | data$installer == 'JAPAN' | data$installer == 'JAPAN EMBASSY' | data$installer == 'Japan Government'] = 'Japan'
data$installer[data$installer == 'Gove' | data$installer == 'Misri Government' | data$installer == 'Gover' | data$installer == 'GOVER' | data$installer == 'GOVERM' | data$installer == 'GOVERN' | data$installer == 'Governme' | data$installer == 'GOVERNME' | data$installer == 'Governmen' | data$installer == 'Government' | data$installer == 'GOVERNMENT' | data$installer == 'Government /Community' | data$installer == 'Government /SDA' | data$installer == 'Government /TCRS' | data$installer == 'Government /world Vision' | data$installer == 'Government and Community' | data$installer == 'Government of Misri' | data$installer == 'Government/TCRS'] = 'Government'
data$installer[data$installer == 'Mission' | data$installer == 'MISSION' | data$installer == 'Missionaries' | data$installer == 'missionary'] = 'Missionary'
data$installer[data$installer == 'Mombo urban  water' | data$installer == 'Mombo urban water' | data$installer == 'Mombo urban water s'] = 'Mombo urban water'
data$installer[data$installer == 'MKONGO BUILDING CONTRACTOR' | data$installer == 'MKON CONSTRUCTION' | data$installer == 'MKONG CONSTRUCTION' | data$installer =='MKONGO CONSTRUCTION'] = 'MKONGO'
data$installer[data$installer == 'Distri' | data$installer == 'Distric Water Department' | data$installer == 'District water depar' | data$installer == 'District water department' | data$installer == 'District Water Department'] = 'District water'
data$installer[data$installer == 'District Council' | data$installer == 'District Counci' | data$installer == 'District council' | data$installer == 'District Council' | data$installer == 'Disrict COUNCIL' | data$installer == 'DISTRICT COUNCIL'] = 'District council'
data$installer[data$installer == 'Moravian' | data$installer == 'Morovi' | data$installer == 'Morovian' | data$installer == 'morovian church' | data$installer == 'Morovian Church' | data$installer == 'Morovian church' | data$installer == 'Morrov' | data$installer == 'Morrovian'] = 'Morovian'
data$installer[data$installer == 'Mosque' | data$installer == 'MOSQUE' | data$installer == 'Mosqure'] = 'Mosque'
data$installer[data$installer == 'Af' | data$installer == 'Africa' | data$installer == 'AFRICA' | data$installer == 'Africa Amini Alama' | data$installer == 'Africa Islamic Agency Tanzania' | data$installer == 'Africa M' | data$installer == 'AFRICA MUSLIM' | data$installe == 'Africa Muslim Agenc' | data$installer == 'AFRICAN DEVELOPMENT FOUNDATION' | data$installer == 'African Muslims Age' | data$installer == 'African Ralief Committe of Ku' | data$installer == 'AFRICAN REFLECTIONS FOUNDATION' | data$installer == 'Africaone' | data$installer == 'Africaone Ltd' | data$installer == 'Africare' | data$installer == 'AGRICAN'] = 'Africa'
data$installer[data$installer == 'MS' | data$installer == 'ms'] = 'Ms'
data$installer[data$installer == 'Msabi' | data$installer == 'MSABI'] = 'Msabi'
data$installer[data$installer == 'MSF/TACARE' | data$installer == 'MSF'] = 'MSF'
data$installer[data$installer == 'kanisa' | data$installer == 'Kanisa' | data$installer == 'Kanisa katoliki' | data$installer == 'Kanisa la TAG' | data$installer == 'Kanisani'] = 'Kanisa'
data$installer[data$installer == 'MSIKIT' | data$installer == 'Msikiti' | data$installer == 'Msiki' | data$installer == 'MISIKITI' | data$installer == 'Msikitini'] = 'Msiki'
data$installer[data$installer == 'IN' | data$installer == 'In' | data$installer == 'India' | data$installer == 'Indi'] = 'India'
data$installer[data$installer == 'Individual' | data$installer == 'INDIVIDUAL' | data$installer == 'Individuals' | data$installer == 'INDIVIDUALS'] = 'Individual'
data$installer[data$installer == 'Institutiona' |data$installer == 'Insititutiona' | data$installer == 'Institution' | data$installer == 'Institutional'] = 'Institutional'
data$installer[data$installer == 'Inter' | data$installer == 'Internal Drainage Basin' | data$installer == 'International Aid Services'] = 'International'
data$installer[data$installer == 'Masjid' | data$installer == 'Masjid Nnre' | data$installer == 'MasjId Takuar'] = 'Masjid'
data$installer[data$installer == 'IS' | data$installer == 'is' | data$installer == 'Is'] = 'Is'
data$installer[data$installer == 'ISF' | data$installer == 'ISF / TASAFF' | data$installer == 'ISF and TACARE' | data$installer =='ISF/Government' | data$installer == 'ISF/TACARE'] = 'ISF'
data$installer[data$installer == 'Islam' | data$installer == 'Islamic' | data$installer == 'Islamic Agency Tanzania' | data$installer == 'Islamic community'] = 'Islam'
data$installer[data$installer == 'ITALI' | data$installer == 'Italian government' | data$installer == 'Italy government'] = 'Italy'
data$installer[data$installer == 'Halmashauri wilaya' | data$installer == 'Halmashauri ya wilaya' | data$installer == 'Halmashauri ya wilaya sikonge'] = 'wilaya'
data$installer[data$installer == 'GRUMENTI' | data$installer == 'GRUMET'| data$installer == 'GURUMETI SAGITA' | data$installer == 'GRUMETI' | data$installer == 'Grumeti fund' | data$installer == 'GRUMETI SINGITA' | data$installer == 'GRUMETI SAGITA' | data$installer == 'GURUMETI SAGITA CO'] = 'Grumeti'
data$installer[data$installer == 'Gwasco' | data$installer == 'Gwasco L' | data$installer == 'Gwaseco'] = 'Gwaseco'
data$installer[data$installer == 'Gtz' | data$installer == 'GTZ' ] = 'Gtz'
data$installer[data$installer == 'Halmashauri ya manispa  tabora' | data$installer == 'Halmashauri ya manispa tabora'] = 'tabora'
data$installer[data$installer == 'Handeni Trunk Main' | data$installer == 'Handeni Trunk Main(' ] = 'Handeni'
data$installer[data$installer == 'Hanja' | data$installer == 'Hanja Lt'] = 'Hanja'
data$installer[data$installer == 'GERMAN' | data$installer == 'GERMAN MISSIONSRY' | data$installer == 'germany' | data$installer == 'Germany' | data$installe == 'GERMANY MISSIONARY'] = 'Germany'
data$installer[data$installer == 'go' | data$installer == 'Go' ] = 'Go'
data$installer[data$installer == 'GLOBAL RESOURCE CO' | data$installer == 'GLOBAL RESOURCE CONSTRUCTION' ] = 'Global resource'
data$installer[data$installer == 'Gold star' | data$installer == 'Goldmain' | data$installer == 'Goldstar'] = 'Gold star'
data$installer[data$installer == 'Grobal resource  alliance' | data$installer == 'Grobal resource alliance'] = 'Grobal'
data$installer[data$installer == 'Fin water' | data$installer == 'Fin Water' | data$installer == 'FIN WATER' | data$installer == 'Fini water' | data$installer == 'Fini Water' | data$installer == 'FiNI WATER' | data$installer == 'FINI Water' | data$installer == 'FINI WATER' | data$installer == 'FINN WATER' | data$installer == 'FinW' |data$installer == 'FinWate' | data$installer == 'Finwater' | data$installer == 'FinWater' ]= 'Fin water'
data$installer[data$installer == 'Ardhi and PET Companies' | data$installer == 'Ardhi Instute' | data$installer == 'Ardhi water well' | data$installer == 'Ardhi Water well' |data$installer == 'Ardhi Water Wells'] = 'Ardhi'
data$installer[data$installer == 'BESADA' | data$installer == 'BESADO'] = 'Besada'
data$installer[data$installer == 'Biore' | data$installer == 'BioRe' | data$installer == 'BIORE'] = 'Biore'
data$installer[data$installer == 'Britain' | data$installer == 'British' | data$installer == 'British colonial government' | data$installer == 'British government'] = 'British'
data$installer[data$installer == 'CARE' | data$installer == 'Care  international' | data$installer == 'care international' | data$installer == 'Care international' | data$installer == 'CARE/CIPRO'] = 'Care'
data$installer[data$installer == 'CARITAS' | data$installer == 'CARTAS' | data$installer == 'CARTAS Tanzania'] = 'Caritas'
data$installer[data$installer == 'CIP' | data$installer == 'CIPRO' | data$installer == 'CIPRO/CARE' | data$installer == 'CPRO' |data$installer == 'CIPRO/CARE/TCRS' | data$installer == 'CIPRO/Government'] = 'CIPRO'
data$installer[data$installer == 'COMMU' | data$installer == 'commu' | data$installer == 'Commu' |data$installer == 'Communit' | data$installer == 'Community' | data$installer == 'COMMUNITY' | data$installer == 'COMMUNITY BANK' | data$installer == 'Comunity'] = 'Community'
data$installer[data$installer == 'Consultant and DWE' | data$installer == 'Consultant' | data$installer == 'Consultant Engineer' | data$installer == 'Consultin engineer' | data$installer == 'Consulting Engineer' | data$installer == 'Consuting Engineer' ] = 'Consulting'
data$installer[data$installer == 'Conta' | data$installer == 'Contr'] = 'Conta'
data$installer[data$installer == 'Cosmo' | data$installer == 'COSMOS ENG LTD' | data$installer == 'Cosmos Engineering'] = 'Cosmos'
data$installer[data$installer == 'coun' | data$installer == 'COUN' | data$installer == 'Counc' | data$installer == 'Council'] = 'Coun'
data$installer[data$installer == 'Ncaa' | data$installer == 'NCAA'] = 'Ncaa'
data$installer[data$installer == 'nchagwa' | data$installer == 'Nchagwa'] = 'Nchagwa'
data$installer[data$installer == 'Nerthlands' | data$installer == 'Netherlands'] = 'wilaya'
data$installer[data$installer == 'NG' | data$installer == 'Ng\'omango' | data$installer == 'Ngelepo group'] = 'Ng'
data$installer[data$installer == 'nandra Construction' | data$installer == 'Nandra Construction' | data$installer == 'NANRA contractor'] = 'Nandra'
data$installer[data$installer == 'Nasan workers' | data$installer == 'Nassan workers' | data$installer == 'Nassor Fehed'] = 'Nassan'
data$installer[data$installer == 'Mzung' | data$installer == 'Mzungu' | data$installer == 'Mzungu Paul'] = 'Mzungu'
data$installer[data$installer == 'mzee mabena' | data$installer == 'Mzee Omari' | data$installer == 'Mzee Salum Bakari Darus' | data$installer == 'Mzee Smith' | data$installer == 'Mzee Waziri Tajari' | data$installer == 'Mzee Yassin Naya'] = 'Mzee'
data$installer[data$installer == 'Naishu construction co. ltd' | data$installer == 'Naishu Construction Co. ltd' | data$installer == 'Naishu construction co.ltd'] = 'Naishu'
data$installer[data$installer == 'Nampapanga' | data$installer == 'Nampopanga' | data$installer == 'Napupanga'] = 'Nampapanga'
data$installer[data$installer == 'Mwalimu  Muhenza' | data$installer == 'Mwalimu  Muhenzi'] = 'Mwalimu'
data$installer[data$installer == 'MWAKI CONTRACTO' | data$installer == 'MWAKI CONTRACTOR' | data$installer == 'mwakifuna'] = 'Mwaki'
data$installer[data$installer == 'mwita' | data$installer == 'mwita kichere' | data$installer == 'mwita Lucas'| data$installer == 'Mwl.Mwita' | data$installer == 'Mwita Machoa' | data$installer == 'Mwita Mahiti' | data$installer == 'Mwita Muremi'] = 'Mwita'
data$installer[data$installer == 'MWL NGASSA' | data$installer == 'Mwl. Nyerere sec. school' | data$installer == 'Mwl. Nyerere sec.school'] = 'Mwl'
data$installer[data$installer == 'not known' | data$installer == 'Not known' | data$installer == 'Not kno'] = 'Desconocido'
data$installer[data$installer == 'No' | data$installer == 'NORA' | data$installer == 'Norad' | data$installer == 'NORAD' | data$installer == 'NORAD/' | data$installer == 'Norani'] = 'Norad'
data$installer[data$installer == 'Ns' | data$installer == 'NSC'] = 'Nsc'
data$installer[data$installer == 'NGO' | data$installer == 'NGO\'s'] = 'NGO'
data$installer[data$installer == 'NYAKILANGANI'| data$installer == 'NYAKILANGANI  CO'| data$installer == 'NYAKILANGANI CONSTRUCTION' |data$installer == 'NYAKILANGANI CONSTRUCTION' | data$installer == 'NYAKILANGANY  CO' | data$intaller == 'NYAKILANGANI CO' | data$installer == 'NYAKILANGANI  CO' | data$installer == 'NYAKILANGANI CONSTRUCTION' | data$installer == 'Nyakilanganyi'| data$installer == 'NYAKILANGANI  CO'| data$installer == 'NYAKILANGANI CO' | data$installer == 'Nyakilanganyi'] = 'Nyaki'
data$installer[data$installer == 'NYAKILANGANI'| data$installer == 'Nyakilanganyi' | data$installer == 'NYAKILANGANI  CO' |data$installer == 'NYAKILANGANI CO' | data$installer == 'NYAKILANGANI CONSTRUCTION'] = 'Nyaki'
data$installer[data$installer == 'MWE' | data$installer == 'MWE &'] = 'MWE'
data$installer[data$installer == 'NGO' | data$installer == 'NGO\'s'] = 'NGO'
data$installer[data$installer == 'Mw' | data$installer == 'MW'] = 'MW'
data$installer[data$installer == 'MUWASA' | data$installer == 'MUWSA' |data$installer == 'Muwaza'] = 'Muwasa'
data$installer[data$installer == 'Municipal' | data$installer == 'Municipal Council'] = 'Municipal'
data$installer[data$installer == 'MUSLEMEHEFEN INTERNATIONAL' | data$installer == 'Muslims' | data$installer == 'Muslimu Society (Shia)'] = 'Musli'
data$installer[data$installer == 'Msiki' | data$installer == 'MSIKITI'] = 'Msiki'
data$installer[data$installer == 'Msabi' | data$installer == 'Msagin' |data$installer == 'Msig'] = 'Msabi'
data$installer[data$installer == 'Mr Chi' | data$installer == 'Mr Kwi' | data$installer == 'Mr Kas'| data$installer == 'Mr Luo' | data$installer == 'Mr Sau' | data$installer =='MREMI CONTRACTOR'] = 'MR'
data$installer[data$installer == 'MP' | data$installer == 'MP Mloka'] = 'MP'
data$installer[data$installer == 'Mpang' | data$installer == 'Mpango wa Mwisa'] = 'Mpang'
data$installer[data$installer == 'MLAKI  CO' | data$installer == 'MLAKI CO'] = 'MLAKI'
data$installer[data$installer == 'Mh Kapuya' | data$installer == 'Mh.chiza'] = 'Mh'
data$installer[data$installer == 'MBULI CO' | data$installer == 'MBULU DISTRICT COUNCIL'] = 'MBuli'
data$installer[data$installer == 'Mama Hamisa' | data$installer == 'Mama Agnes Kagimbo' | data$installer == 'Mama joela' | data$installer == 'Mama Kalage' | data$installer == 'Mama Kapwapwa'] = 'Mama'
data$installer[data$installer == 'Maji' | data$installer == 'Maji block' | data$installer == 'Maji mugumu'] = 'Maji'
data$installer[data$installer == 'O' | data$installer == 'O &'] = 'O'
data$installer[data$installer == 'Oikos e .Africa' |data$installer =='OIKOS'| data$installer == 'Oikos E .Africa'| data$installer == 'Oikos E Africa' | data$installer == 'Oikos E. Africa' | data$installer == 'Oikos E.Africa' | data$installer == 'Oikos E.Afrika'] = 'Oikos'
data$installer[data$installer == 'Jeshi la wokovu [cida]' | data$installer == 'Jeshi la wokovu' | data$installer == 'JESHI LA WOKOVU' | data$installer == 'Jeshi la Wokovu'] = 'Jeshi'
data$installer[data$installer == 'JHL CO LTD' | data$installer == 'JLH CO LTD' | data$installer == 'J LH CO LTD'] = 'JHL'
data$installer[data$installer == 'Hospi' | data$installer == 'Hospital'] = 'Hospital'
data$installer[data$installer == 'HOTELS AND LOGGS TZ LTD' | data$installer == 'HOTEL AND LODGE TANZANIA'] = 'HOTEL'
data$installer[data$installer == 'HOWARD HUMFREYS' | data$installer == 'Humfreys Co' | data$installer == 'Howard and humfrey consultant' | data$installer == 'Howard and Humfrey Consultants'] = 'Howard'
data$installer[data$installer == 'Heri mission' | data$installer == 'Hery'] = 'Hery'
data$installer[data$installer == 'Hemed Abdalkah' | data$installer == 'Hemed Abdallah'] = 'Hemed'
data$installer[data$installer == 'HAPA SINGIDA' | data$installer == 'HAPA'] = 'HAPA'
data$installer[data$installer == 'Halmashauri ya mburu' | data$installer == 'Halmashauri/Quick win project'] = 'Halmashauri'
data$installer[data$installer == 'Greec' | data$installer == 'Green'] = 'Green'
data$installer[data$installer == 'GREINEKER' | data$installer == 'GREINAKER'] = 'GREINEKER'
data$installer[data$installer == 'GRA TZ MUSOMA' | data$installer == 'GRA'] = 'GRA'
data$installer[data$installer == 'George' | data$installer == 'George mtoto' | data$installer == 'George mtoto company'] = 'George'
data$installer[data$installer == 'G.D&I.D' | data$installer == 'GD&ID'] = 'GDID'
data$installer[data$installer == 'FPCT' | data$installer == 'FPTC' | data$installer == 'FPCT Church'] = 'FPTC'
data$installer[data$installer == 'Ox' | data$installer == 'OXFAM' | data$installer == 'OXFARM'] = 'Oxfarm'
data$installer[data$installer == 'p' | data$installer == 'P'] = 'P'
data$installer[data$installer == 'PAD' | data$installer == 'Padep' | data$installer == 'PADEP'] = 'Padep'
data$installer[data$installer == 'Onesm' | data$installer == 'ONESM'] = 'Onesm'
data$installer[data$installer == 'Pentecosta' | data$installer == ' Free Pentecoste Church of Tanz' | data$installer == 'Pentecost church' | data$installer == 'Pentecostal church' | data$installer == 'Pentekoste'] = 'Pentecostal'
data$installer[data$installer == 'Oldadi village community'| data$installer == 'Oldadai village community'| data$installer == 'Olgilai village community'] = 'Ovc'
data$installer[data$installer == 'MTUWASA and Community' | data$installer == 'MTUWASA'] = 'MTUWASA'
data$installer[data$installer == 'PET'| data$installer == 'Pet Corporation Ltd'| data$installer == 'Pet  Corporation  Ltd'| data$installer == 'Pet Coporation Ltd' | data$installer == 'Pet Corporation  Ltd' | data$installer == 'pet Corporation Ltd' | data$installer == 'Pet corporation Ltd'] = 'Pet'
data$installer[data$installer == 'peter' | data$installer == 'Peter Mayiro' | data$installer == 'Petro Patrice'] = 'Peter'
data$installer[data$installer == 'Juma' | data$installer == 'Juma Makulilo' | data$installer == 'Juma Maro' | data$installer == 'Juma Ndege' | data$installer == 'Jumaa' | data$installer == 'Jumanne' | data$installer == 'Jumanne Siabo'] = 'Juma'
data$installer[data$installer == 'plan int' | data$installer == 'plan Int' | data$installer == 'Plan Int' | data$installer == 'Plan Internationa'| data$installer == 'Plan International'] = 'Plan international'
data$installer[data$installer == 'People form Egypt' | data$installer == 'People P'] = 'People'
data$installer[data$installer == 'COEK' | data$installer == 'COEW'] = 'COEK'
data$installer[data$installer == 'Colonial Government' | data$installer == 'Colonial government'] = 'Colonial government'
data$installer[data$installer == 'Compa' | data$installer == 'Company'] = 'Company'
data$installer[data$installer == 'Consulting engineer' | data$installer == 'Consulting'] = 'Consulting'
data$installer[data$installer == 'CJEJOW CONSTRUCTION' | data$installer == 'CJEJ0' | data$installer == 'CJEJOW'] = 'CJE'
data$installer[data$installer == 'Enyueti' | data$installer == 'Enyuati'] = 'Enyuati'
data$installer[data$installer == 'EGYPT REGWA' | data$installer == 'Egypt Technical Co Operation' | data$installer == 'Egypst Government' | data$installer == 'EGYPT'] = 'Egypt'
data$installer[data$installer == 'DSP' | data$installer == 'DSPU'] = 'DSP'
data$installer[data$installer == 'Dr. Matobola' | data$installer == 'DR. Matomola' | data$installer == 'Dr.Matobola' | data$installer == 'Dr.Matomola'] = 'Drmato'
data$installer[data$installer == 'Dmdd' | data$installer == 'DMDD' | data$installer == 'DMDD/SOLIDER' | data$installer == 'DMK' | data$installer == 'DMMD'] = 'DMDD'
data$installer[data$installer == 'Do' | data$installer == 'DO'] = 'Do'
data$installer[data$installer == 'DESK a' | data$installer == 'DESK A' | data$installer == 'desk and chair foundation' | data$installer == 'Desk and chair foundation' | data$installer == 'DESK C'] = 'Desk'
data$installer[data$installer == 'District Council' | data$installer == 'District council' | data$installer == 'District COUNCIL' ] = 'District council'
data$installer[data$installer == 'DDSA' | data$installer == 'DDCA' | data$installer == 'DDCA CO' | data$installer == 'DCCA'] = 'DCCA'
data$installer[data$installer == 'PRIV' | data$installer == 'Priva' | data$installer == 'Privat' | data$installer == 'private' | data$installer == 'Private' | data$installer == 'Private company' | data$installer == 'Private individuals' | data$installer == 'PRIVATE INSTITUTIONS' | data$installer == 'Private owned' | data$installer == 'Private person' | data$installer == 'Private Technician'] = 'Drmato'
data$installer[data$installer == 'Quick win project' | data$installer == 'Qwick Win' | data$installer == 'Quick win projet /Council'| data$installer == 'Quick win project /Council' | data$installer == 'Quick win/halmashauri' | data$installer == 'QUICKWINS' | data$installer == 'Quik' | data$installer == 'QUIK' | data$installer == 'QUKWIN' | data$installer == 'QUWKWIN' | data$installer == 'Qwick win' | data$installer == 'QWICKWIN'] = 'QW'
data$installer[data$installer == 'R' | data$installer == 'RCchurch/CEFA' | data$installer == 'RC Msufi' | data$installer == 'RC Njoro' | data$installer == 'RC/Mission' | data$installer == 'RC mission' | data$installer == 'RC MISSION' | data$installer == 'RC MISSIONARY'| data$installer == 'RC Mi' | data$installer == 'RC Mis' | data$installer == 'Rc Mission' | data$installer == 'RC Church' | data$installer == 'RC CHURCH' | data$installer == 'RC CHURCH BROTHER' | data$installer == 'RC church/CEFA' | data$installer == 'RC church/Central Gover' |data$installer == 'RC church'| data$installer == 'rc church' | data$installer == 'RC CH' | data$installer == 'RC Churc' | data$installer == 'RC CATHORIC' | data$installer == 'rc ch' | data$installer == 'RC Ch'| data$installer == 'RC' | data$installer == 'RC .Church' | data$installer == 'RC C'|data$installer == 'R.C' | data$installer == 'Rc'] = 'R'
data$installer[data$installer == 'Rashid Mahongwe' | data$installer == 'Rashid Seng\'ombe'] = 'Rashid'
data$installer[data$installer == 'RDC' | data$installer == 'RDDC'] = 'RDC'
data$installer[data$installer == 'Ramadhani M. Mvugalo' | data$installer == 'Ramadhani Nyambizi' ] = 'Ramad'
data$installer[data$installer == 'ISSAA KANYANGE' | data$installer == 'ISSAC MOLLEl' | data$installer == 'ISSAC MOLLEL' | data$installer == 'Issa mohamedi Tumwanga'] = 'ISSA'
data$installer[data$installer == 'Ir' | data$installer == 'ir'] = 'Ir'
data$installer[data$installer == 'GACHUMA CONSTRUCTION' | data$installer == 'GACHUMA GINERY'] = 'GACHUMA'
data$installer[data$installer == 'FARM-AFRICA' | data$installer == 'Farm Africa'] = 'Farm africa'
data$installer[data$installer == 'EMANDA' | data$installer == 'EMANDA BUILBERS'] = 'EMANDA'
data$installer[data$installer == 'Diwani' | data$installer == 'DIWANI'] = 'Diwani'
data$installer[data$installer == 'DHV' | data$installer == 'DHV Moro'] = 'DHV'
data$installer[data$installer == 'Dawasco' | data$installer == 'DAWASCO'] = 'Dawasco'
data$installer[data$installer == 'DD ES SALAAM ROUND TABLE' | data$installer == 'Dar es salaam Technician'] = 'Dar'
data$installer[data$installer == 'DANIAD' | data$installer == 'DANIDA' | data$installer == 'DANIDA CO' | data$installer == 'DANNIDA' | data$installer == 'DANIDS' | data$installer == 'DANID' | data$installer == 'Danid'] = 'DANIDA'
data$installer[data$installer == 'REDEP' | data$installer == 'REDAP' | data$installer == 'Redep'] = 'REDP'
data$installer[data$installer == 'Region water' | data$installer == 'regwa Company' | data$installer == 'REGWA Company' | data$installer == 'Regwa Company' | data$installer == 'REGWA COMPANY OF EGPTY' |data$installer == 'REGWA COMPANY OF EGYPTY' | data$installer == 'REGWA COMPANY OF EGYPT' | data$installer == 'Region water Department' | data$installer == 'Region Water Department' | data$installer == 'Regional Water' | data$installer == 'REGIONAL WATER ENGINEER ARUSHA' | data$installer == 'REGWA'] = 'REGWA'
data$installer[data$installer == 'Red cross' | data$installer == 'Red Cross' | data$installer == 'RED CROSS'] = 'Red cross'
data$installer[data$installer == 'Resolute' | data$installer == 'RESOLUTE MINING'] = 'Resolute'
data$installer[data$installer == 'Rhoda' | data$installer == 'Rhodi' | data$installer == 'Rhodi Wamburs' | data$installer == 'Rhobi' | data$installer == 'Rhobi Wamburs'] = 'Rhodi'
data$installer[data$installer == 'Abdallah Ally Wazir' | data$installer == 'ABDALA' ] = 'ABDALA'
data$installer[data$installer == 'ACTIVE TANK CO' | data$installer == 'ACTIVE TANK CO LDT' ] = 'Active'
data$installer[data$installer == 'ADB' | data$installer == 'ADP' | data$installer == 'ADP Bsusangi'] = 'ADP'
data$installer[data$installer == 'AI' | data$installer == 'AIC' | data$installer == 'AIC KI' | data$installer == 'AICT'] = 'AIC'
data$installer[data$installer == 'Amari' | data$installer == 'Amadi'] = 'Amadi'
data$installer[data$installer == 'Amboni plantation' | data$installer == 'Amboni Plantation'] = 'Amboni'
data$installer[data$installer == 'AMP Contract' | data$installer == 'AMP contractor' | data$installer == 'AMp Contracts'] = 'AMP'
data$installer[data$installer == 'Amref' | data$installer == 'AMREF'] = 'AMREF'
data$installer[data$installer == 'Angli' | data$installer == 'Anglican' | data$installer == 'ANGLI' | data$installer == 'Anglica Church' | data$installer == 'Anglican' |data$installer == ' Anglican Church' | data$installer == 'anglican Uganda' | data$installer == 'Anglican Uganda' | data$installer == 'Anglikan' | data$installer == 'Anglikana' | data$installer == 'ANGLIKANA CHURCH' | data$installer == 'Angrikana' | data$installer == 'ANGRIKANA'] = 'Anglicana'
data$installer[data$installer == 'AQUAL' | data$installer == 'AQUA BLUES ANGELS' | data$installer == 'AQUA Wat' | data$installer == 'AQUA Wel' | data$installer == 'AQUA WEL' | data$instaler == 'Aqual' | data$installer == 'AQUARMAN DRILLERS' | data$installer == 'Aqwaman Drilling'] = 'AQUA'
data$installer[data$installer == 'Atlas' | data$installer == 'Atlas Company'] = 'Atlas'
data$installer = as.factor(data$installer)

data$permit = as.character(data$permit)
data$permit[data$permit == ''] = 'True'
data$permit = as.factor(data$permit)

data$scheme_management = as.character(data$scheme_management)
data$scheme_management[data$scheme_management == '' | data$scheme_management == 'None'] = 'Other'
data$scheme_management = as.factor(data$scheme_management)

data$public_meeting = as.character(data$public_meeting)
data$public_meeting[data$public_meeting == ''] = 'True'
data$public_meeting = as.factor(data$public_meeting)

data$installer = as.character(data$installer)
data$installer[data$installer == 0 | data$installer == '-'] = 'Other'
data$installer = as.factor(data$installer)

data$gps_height[data$gps_height == 0] = 1166
data$date_recorded = as.Date(data$date_recorded)
data$extraction_type_group = NULL

data$extraction_type = as.character(data$extraction_type)
data$extraction_type[data$extraction_type == 'india mark ii'] = 'india'
data$extraction_type[data$extraction_type == 'india mark iii'] = 'india'
data$extraction_type[data$extraction_type == 'other - swn 81' | data$extraction_type == 'swn 80'] = 'swn'
data$extraction_type[data$extraction_type == 'walimi' | data$extraction_type == 'other - mkulima/shinyanga' | data$extraction_type == 'other - play pump'] = 'other handpump'
data$extraction_type[data$extraction_type == 'cemo' | data$extraction_type == 'climax'] = 'other motorpump'
data$extraction_type = as.factor(data$extraction_type)

inicio_test = nrow(train) + 1
fin = nrow(data)
train = data[1:(inicio_test-1),]
test = data[inicio_test:fin,]



#model.Ripper25 = JRip(status_group~latitude+longitude+date_recorded+basin+lga+funder+population+construction_year+installer+
 #                       gps_height+public_meeting+scheme_management+permit+extraction_type+management+
  #                      management_group+payment+quality_group+quantity+source+ source_class+
   #                     waterpoint_type, train, control = Weka_control(F = 2, N=3,O=29))

#summary(model.Ripper25)
#model.Ripper25.pred = predict(model.Ripper25,newdata = test)

#generaSubida('25',test$id,model.Ripper25.pred)

model.Ripper28 = JRip(status_group~date_recorded+basin+lga+funder+population+construction_year+installer+
                        gps_height+public_meeting+scheme_management+permit+extraction_type+management+
                        management_group+payment+quality_group+quantity+source+
                        waterpoint_type, train, control = Weka_control(F = 2, N=3,O=29))

summary(model.Ripper28)
model.Ripper28.pred = predict(model.Ripper28,newdata = test)

generaSubida('28',test$id,model.Ripper28.pred)

