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

antigua_subida = read.csv("new.csv")
test = cbind(test, status_group=antigua_subida$status_group)

test$construction_year[test$construction_year == 0 & test$status_group == 'functional'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'functional']))
test$construction_year[test$construction_year == 0 & test$status_group == 'non functional'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'non functional']))
test$construction_year[test$construction_year == 0 & test$status_group == 'functional needs repair'] = round(mean(test$construction_year[test$construction_year != 0 & test$status_group == 'functional needs repair']))


# Creación de la variable estado en el test para que
# coincidan en número a la hora de hacer transformaciones
test$status_group = ""

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


model.Ripper28 = JRip(status_group~date_recorded+basin+lga+funder+population+construction_year+installer+
                        gps_height+public_meeting+scheme_management+permit+extraction_type+management+
                        management_group+payment+quality_group+quantity+source+
                        waterpoint_type, train, control = Weka_control(F = 2, N=3,O=29))

summary(model.Ripper28)
model.Ripper28.pred = predict(model.Ripper28,newdata = test)

generaSubida('28',test$id,model.Ripper28.pred)
