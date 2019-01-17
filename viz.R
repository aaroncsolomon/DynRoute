require(ggplot2)
require(data.table)

l11 <- data.table(t(fread('l11.csv')))
setnames(l11, c("Epoch", "Label"))

ggplot(data=l11, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1))


## for early output graphs
e1 <- data.table(t(fread('received_data/early_c1.csv')))
e2 <- data.table(t(fread('received_data/early_c2.csv')))
e3 <- data.table(t(fread('received_data/early_c3.csv')))

setnames(e1, c("Epoch", "Label"))
setnames(e2, c("Epoch", "Label"))
setnames(e3, c("Epoch", "Label"))

ggplot(data=e1, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Early Output 1")

ggplot(data=e2, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Early Output 2")

ggplot(data=e3, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Main Output")



## for interlayer routing 

i1 <- data.table(t(fread('received_data/interlayer_l11.csv')))
i2 <- data.table(t(fread('received_data/interlayer_l12.csv')))
i3 <- data.table(t(fread('received_data/interlayer_c3.csv')))

setnames(i1, c("Epoch", "Label"))
setnames(i2, c("Epoch", "Label"))
setnames(i3, c("Epoch", "Label"))

ggplot(data=i1, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Interlayer Conv1 -> Linear")

ggplot(data=i2, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Interlayer Conv2 -> Linear")

ggplot(data=i3, aes(x=Label, fill=factor(Label))) + geom_bar() +
  facet_wrap(~Epoch) + scale_x_discrete(limits=seq(0,9,1)) + ggtitle("Interlayer Conv1 -> Conv3")


### looking at times 

bas_time  <- data.table(t(fread('basic_times.csv')))
setnames(bas_time, c("Epoch", "Time"))
bas_time[, Partition:="All Layers"]
bas_time[, Network:="Simple 3 Layer"]


## interlayer times 

lt1 <- data.table(t(fread('time_interlayer_l1.csv')))
lt2 <- data.table(t(fread('time_interlayer_l2.csv')))
dt1 <- data.table(t(fread('time_interlayer_d1.csv')))
dt2 <- data.table(t(fread('time_interlayer_d2.csv')))
outt <- data.table(t(fread('time_interlayer_ot.csv')))

setnames(lt1, c("Epoch", "Time"))
setnames(lt2, c("Epoch", "Time"))
setnames(dt1, c("Epoch", "Time"))
setnames(dt2, c("Epoch", "Time"))
setnames(outt, c("Epoch", "Time"))

lt1[, Partition := 'Conv Layer 1']
lt2[, Partition := 'Conv Layer 2']
dt1[, Partition := 'Decision Layer 1']
dt2[, Partition := 'Decision Layer 2']
outt[, Partition := 'Conv 3 and Output']

interlayer_timings <- rbind(lt1, lt2, dt1, dt2, outt)
interlayer_timings[, Network :=  "Interlayer"]

## early output times 

lt1 <- data.table(t(fread('time_early_l1.csv')))
lt2 <- data.table(t(fread('time_early_l2.csv')))
dt1 <- data.table(t(fread('time_early_d1.csv')))
dt2 <- data.table(t(fread('time_early_d2.csv')))
outt <- data.table(t(fread('time_early_ot.csv')))

setnames(lt1, c("Epoch", "Time"))
setnames(lt2, c("Epoch", "Time"))
setnames(dt1, c("Epoch", "Time"))
setnames(dt2, c("Epoch", "Time"))
setnames(outt, c("Epoch", "Time"))

lt1[, Partition := 'Conv Layer 1']
lt2[, Partition := 'Conv Layer 2']
dt1[, Partition := 'Decision Layer 1 and Output']
dt2[, Partition := 'Decision Layer 2 and Output']
outt[, Partition := 'Conv 3 and Output']

early_timings <- rbind(lt1, lt2, dt1, dt2, outt)
early_timings[, Network :=  "Early Output"]


dataa <- rbind(bas_time, interlayer_timings, early_timings)

dataa[, Partition := factor(Partition, levels = c('Conv Layer 1', 'Decision Layer 1 and Output', 'Decision Layer 1', 'Conv Layer 2', 'Decision Layer 2 and Output', 'Decision Layer 2', 'Conv 3 and Output', 'All Layers'))]

ggplot(data= dataa, aes(x=Partition, color=factor(Network), y = Time)) + 
  geom_jitter() + facet_wrap(~Network, scales = 'free_x') + theme(axis.text.x = element_text(angle = 90))

mem <- fread('~/Desktop/memory.csv')
mem[, Network := factor(Network, levels = c('Interlayer', 'Early Output', 'Simple'))]

ggplot(data =mem, aes(x=Network, y=Memory, color = factor(Network))) + 
  geom_boxplot()

