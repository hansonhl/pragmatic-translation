import torch
import time
from translators.summary import NaiveSummary, OptimizedSummary, greedy_summary
from translators.translators import Unfold, Constant

s = OptimizedSummary('../myOpenNMT')

sent = "-lrb- cnn -rrb- the palestinian authority officially became the 123rd member of the international criminal court on wednesday , a step that gives the court jurisdiction over alleged crimes in palestinian territories . the formal accession was marked with a ceremony at the hague , in the netherlands , where the court is based . the palestinians signed the icc 's founding rome statute in january , when they also accepted its jurisdiction over alleged crimes committed `` in the occupied palestinian territory , including east jerusalem , since june 13 , 2014 . '' later that month , the icc opened a preliminary examination into the situation in palestinian territories , paving the way for possible war crimes investigations against israelis . as members of the court , palestinians may be subject to counter-charges as well . israel and the united states , neither of which is an icc member , opposed the palestinians ' efforts to join the body . but palestinian foreign minister riad al-malki , speaking at wednesday 's ceremony , said it was a move toward greater justice . `` as palestine formally becomes a state party to the rome statute today , the world is also a step closer to ending a long era of impunity and injustice , '' he said , according to an icc news release . `` indeed , today brings us closer to our shared goals of justice and peace . '' judge kuniko ozaki , a vice president of the icc , said acceding to the treaty was just the first step for the palestinians . `` as the rome statute today enters into force for the state of palestine , palestine acquires all the rights as well as responsibilities that come with being a state party to the statute . these are substantive commitments , which can not be taken lightly , '' she said . rights group human rights watch welcomed the development . `` governments seeking to penalize palestine for joining the icc should immediately end their pressure , and countries that support universal acceptance of the court 's treaty should speak out to welcome its membership , '' said balkees jarrah , international justice counsel for the group . `` what 's objectionable is the attempts to undermine international justice , not palestine 's decision to join a treaty to which over 100 countries around the world are members . '' in january , when the preliminary icc examination was opened , israeli prime minister benjamin netanyahu described it as an outrage , saying the court was overstepping its boundaries . the united states also said it `` strongly '' disagreed with the court 's decision . `` as we have said repeatedly , we do not believe that palestine is a state and therefore we do not believe that it is eligible to join the icc , '' the state department said in a statement . it urged the warring sides to resolve their differences through direct negotiations . `` we will continue to oppose actions against israel at the icc as counterproductive to the cause of peace , '' it said . but the icc begs to differ with the definition of a state for its purposes and refers to the territories as `` palestine . '' while a preliminary examination is not a formal investigation , it allows the court to review evidence and determine whether to investigate suspects on both sides . prosecutor fatou bensouda said her office would `` conduct its analysis in full independence and impartiality . '' the war between israel and hamas militants in gaza last summer left more than 2,000 people dead . the inquiry will include alleged war crimes committed since june . the international criminal court was set up in 2002 to prosecute genocide , crimes against humanity and war crimes . cnn 's vasco cotovio , kareem khadder and faith karimi contributed to this report ."

"""
print('-- Starting greedy summary')
start_time = time.time()
pred = greedy_summary(s, sent)
duration = time.time() - start_time
print('-- Summary ended, duration: {:.4}s. Result:\n'.format(duration))
print(pred)
"""

# output
"""
<t> the palestinian authority is the 123rd member of the international criminal
court . </t> <t> the icc is based in the netherlands , paving the way for
possible war crimes investigations . </t>
"""

sent2 = "marseille , france -lrb- cnn -rrb- the french prosecutor leading an investigation into the crash of germanwings flight 9525 insisted wednesday that he was not aware of any video footage from on board the plane . marseille prosecutor brice robin told cnn that `` so far no videos were used in the crash investigation . '' he added , `` a person who has such a video needs to immediately give it to the investigators . '' robin 's comments follow claims by two magazines , german daily bild and french paris match , of a cell phone video showing the harrowing final seconds from on board germanwings flight 9525 as it crashed into the french alps . all 150 on board were killed . paris match and bild reported that the video was recovered from a phone at the wreckage site . the two publications described the supposed video , but did not post it on their websites . the publications said that they watched the video , which was found by a source close to the investigation . `` one can hear cries of ` my god ' in several languages , '' paris match reported . `` metallic banging can also be heard more than three times , perhaps of the pilot trying to open the cockpit door with a heavy object . towards the end , after a heavy shake , stronger than the others , the screaming intensifies . then nothing . '' `` it is a very disturbing scene , '' said julian reichelt , editor-in-chief of bild online . an official with france 's accident investigation agency , the bea , said the agency is not aware of any such video . lt. col. jean-marc menichini , a french gendarmerie spokesman in charge of communications on rescue efforts around the germanwings crash site , told cnn that the reports were `` completely wrong '' and `` unwarranted . '' cell phones have been collected at the site , he said , but that they `` had n't been exploited yet . '' menichini said he believed the cell phones would need to be sent to the criminal research institute in rosny sous-bois , near paris , in order to be analyzed by specialized technicians working hand-in-hand with investigators . but none of the cell phones found so far have been sent to the institute , menichini said . asked whether staff involved in the search could have leaked a memory card to the media , menichini answered with a categorical `` no . '' reichelt told `` erin burnett : outfront '' that he had watched the video and stood by the report , saying bild and paris match are `` very confident '' that the clip is real . he noted that investigators only revealed they 'd recovered cell phones from the crash site after bild and paris match published their reports . `` that is something we did not know before . ... overall we can say many things of the investigation were n't revealed by the investigation at the beginning , '' he said . what was mental state of germanwings co-pilot ? german airline lufthansa confirmed tuesday that co-pilot andreas lubitz had battled depression years before he took the controls of germanwings flight 9525 , which he 's accused of deliberately crashing last week in the french alps . lubitz told his lufthansa flight training school in 2009 that he had a `` previous episode of severe depression , '' the airline said tuesday . email correspondence between lubitz and the school discovered in an internal investigation , lufthansa said , included medical documents he submitted in connection with resuming his flight training . the announcement indicates that lufthansa , the parent company of germanwings , knew of lubitz 's battle with depression , allowed him to continue training and ultimately put him in the cockpit . lufthansa , whose ceo carsten spohr previously said lubitz was 100 % fit to fly , described its statement tuesday as a `` swift and seamless clarification '' and said it was sharing the information and documents -- including training and medical records -- with public prosecutors . spohr traveled to the crash site wednesday , where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside . he saw the crisis center set up in seyne-les-alpes , laid a wreath in the village of le vernet , closer to the crash site , where grieving families have left flowers at a simple stone memorial . menichini told cnn late tuesday that no visible human remains were left at the site but recovery teams would keep searching . french president francois hollande , speaking tuesday , said that it should be possible to identify all the victims using dna analysis by the end of the week , sooner than authorities had previously suggested . in the meantime , the recovery of the victims ' personal belongings will start wednesday , menichini said . among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board . check out the latest from our correspondents . the details about lubitz 's correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and lubitz 's possible motive for downing the jet . a lufthansa spokesperson told cnn on tuesday that lubitz had a valid medical certificate , had passed all his examinations and `` held all the licenses required . '' earlier , a spokesman for the prosecutor 's office in dusseldorf , christoph kumpa , said medical records reveal lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot 's license . kumpa emphasized there 's no evidence suggesting lubitz was suicidal or acting aggressively before the crash . investigators are looking into whether lubitz feared his medical condition would cause him to lose his pilot 's license , a european government official briefed on the investigation told cnn on tuesday . while flying was `` a big part of his life , '' the source said , it 's only one theory being considered . another source , a law enforcement official briefed on the investigation , also told cnn that authorities believe the primary motive for lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems . lubitz 's girlfriend told investigators he had seen an eye doctor and a neuropsychologist , both of whom deemed him unfit to work recently and concluded he had psychological issues , the european government official said . but no matter what details emerge about his previous mental health struggles , there 's more to the story , said brian russell , a forensic psychologist . `` psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they were n't going to keep doing their job and they 're upset about that and so they 're suicidal , '' he said . `` but there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person 's problems . '' germanwings crash compensation : what we know . who was the captain of germanwings flight 9525 ? cnn 's margot haddad reported from marseille and pamela brown from dusseldorf , while laura smith-spark wrote from london . cnn 's frederik pleitgen , pamela boykoff , antonia mortensen , sandrine amiel and anna-maja rappard contributed to this report ."

"""
print('-- Starting greedy summary')
start_time = time.time()
pred = greedy_summary(s, sent2)
duration = time.time() - start_time
print('-- Summary ended, duration: {:.4}s. Result:\n'.format(duration))
print(pred)
"""

# Try with starting model with incomplete sequence
seq = '<t> the'
pred = ''

"""
while pred != s.eos_token:
    log_probs, support = s.forward(seq, sent)
    predid = torch.argmax(log_probs)
    pred = support[predid]
    if pred != s.eos_token:
        if seq == '':
            seq = pred
        else:
            seq = seq + ' ' + pred
print(seq)
"""


print('Now trying unfold')
s0_sent = Unfold(s, beam_width=1)
probs, support = s0_sent.forward(sequence=[],source_sentence=sent,debug=False)
print(support)
sentences = [sent, sent2]

l1_sent = Constant(support=sentences, unfolded_model=s0_sent,change_direction=True)
log_probs, support = l1_sent.forward(sequence=[],source_sentence=sentences[0],debug=False)
print('log_probs:', log_probs)
print('support:', support)


# print(seq)

# ideas for distractor generation
# wordnet
# dependency parsing
# get attention, mask and generate/swap
