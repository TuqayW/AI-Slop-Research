from pathlib import Path
import re
articles = {
"ai01_train_electricity.txt": """THE DISTRIBUTION OF ELECTRIC LIGHT.

The increasing use of electricity for illumination has rendered necessary a
careful consideration of the manner in which the current shall be distributed.
It is one thing to produce the current at a central station and quite another
to convey it economically to lamps situated at different distances. The wires
must be of sufficient capacity to carry the required current without excessive
loss, while at the same time the expense of metal must be kept within reasonable
limits.

In a large installation the demand is not constant. Some lamps are lighted
early in the evening and others later, while certain workshops may require
power during the day. A distribution system must therefore accommodate changes
of load without unnecessary waste. The arrangement of conductors, transformers
and switches becomes almost as important as the generating machinery itself.

It is sometimes supposed that the perfection of the lamp is the chief matter
to be considered. In practice the whole system must be regarded as an
interdependent mechanism. A lamp of excellent construction cannot compensate
for losses in transmission, nor can a powerful generator make a badly planned
network economical. The future development of electric lighting will depend
upon improvements in the system as a whole rather than upon a single invention.""",

"ai02_train_railway_signals.txt": """RAILWAY SIGNALS AND SAFETY.

The rapid extension of railway traffic has made the proper control of trains a
matter of increasing importance. A line upon which only a few trains are
running may be operated with simple arrangements, but the same method becomes
dangerous when the number of movements is greatly increased. The signal then
serves as the visible expression of a set of rules by which one movement is
permitted and another prohibited.

The mechanical signal itself is comparatively simple. The difficulty lies in
determining when it shall change and in making certain that the operator's
action corresponds with the actual condition of the line. A signal which
indicates safety while a train is already occupying the section beyond it is
worse than useless. The apparatus must therefore be connected with the
movements which it is intended to govern.

Automatic arrangements promise assistance in this direction because they
reduce the number of separate decisions required from the signalman. Yet
automatic machinery introduces its own conditions. Contacts may become
imperfect, wires may fail, and mechanical parts require inspection. A good
railway signal system is consequently not merely an ingenious collection of
devices. It is a method of maintaining definite knowledge of where trains are,
which lines are clear, and what movement may safely follow another.""",

"ai03_train_water_pumps.txt": """IMPROVEMENTS IN WATER PUMPS.

The ordinary pumping engine appears at first sight to be a simple machine.
Water enters one part of the apparatus, pressure is applied to it, and the
fluid is delivered through a pipe. Yet the useful effect depends upon many
details which are easily neglected. Valves must open and close at the proper
instant, the passages must be sufficiently large, and the moving parts must
work without excessive friction.

A pump which gives a satisfactory result under a nearly constant supply may
behave quite differently when the level of the water changes. The suction
height then becomes an important consideration, and imperfections in the
valves may produce a marked reduction of delivery. The temptation to judge a
pump merely by the force observed at its outlet is therefore misleading.

There is also an advantage in arranging the working parts so that inspection
can be made without dismantling the entire machine. An apparatus which is
theoretically efficient but troublesome to maintain may prove inferior to a
somewhat simpler construction. The best pumping machinery is likely to be that
which combines regular action with accessibility.

As in many branches of mechanical engineering, improvements do not necessarily
consist in making the machine more elaborate. A better relation among the
existing parts may accomplish more than the addition of another complicated
device.""",

"ai04_train_telegraph_cables.txt": """UNDERGROUND TELEGRAPH CABLES.

The proposal to place telegraph conductors beneath the surface has advantages
which are easily recognized from the appearance of a modern city. Overhead
wires multiply as the population increases, and the poles required to support
them occupy valuable space while adding little to the beauty of the streets.
Yet the removal of the wires does not remove the engineering difficulty.

A buried conductor must be protected against moisture, mechanical injury and
the disturbance produced by excavation. When a fault occurs in an overhead
line, the defective portion may often be discovered by ordinary inspection.
The underground cable conceals its condition and may therefore require a
different method of examination.

The expense of construction is also an important matter. The first cost is
greater because trenches must be prepared and suitable coverings provided.
On the other hand, the finished system is less exposed to wind and accidental
contact. Whether the change is economical must consequently be determined over
the whole period of service.

The important question is not whether one arrangement looks better than the
other, but which system provides the most reliable communication for the money
expended. A practical comparison must include construction, maintenance,
repairs and the interruptions caused by failures. Only such a comparison can
justify a general adoption of underground lines.""",

"ai05_train_steam_engine.txt": """THE GOVERNING OF STEAM ENGINES.

The successful operation of a steam engine requires more than the production
of pressure within the boiler. The engine must receive steam in such a manner
that the power developed remains reasonably uniform under changing loads.
Without some means of regulation, a reduction in the demand may cause the
speed to increase, while an increase in the work required may produce an
undesirable slowing of the machinery.

The governor performs this regulating function by responding to variations in
speed and altering the admission of steam. Its action is therefore indirect:
it does not create the power, but modifies the conditions under which the
engine receives it. The simplicity of this principle should not lead to the
supposition that every governor will act equally well.

The response must be sufficiently rapid without producing continuous
oscillation. A mechanism which closes the admission too suddenly may cause an
irregular motion, while one which moves too slowly may permit the speed to
change considerably before correction occurs. Friction and looseness in the
connecting parts introduce further difficulties.

The practical engineer consequently judges a governing apparatus not merely by
its ability to alter the steam supply, but by the steadiness of the whole
engine while the load varies. A successful arrangement is one whose corrections
are prompt, moderate and repeatable.""",

"ai06_train_telephone_exchange.txt": """THE ORGANIZATION OF A TELEPHONE EXCHANGE.

The rapid growth of telephone service has made the exchange an important
mechanism in its own right. When only a small number of subscribers are
connected, the operator may establish a communication with little difficulty.
As the number increases, however, the possible combinations multiply rapidly,
and the exchange must provide an orderly means of making one connection while
preserving many others.

Each subscriber requires a line leading to the central office, together with
some means of indicating that service is requested. The operator then connects
the calling line with the line belonging to the desired subscriber. The
apparatus must make the connection secure without requiring an excessive number
of independent operations.

The chief difficulty is not the principle of connection but the management of
large numbers of simultaneous calls. A good exchange must provide sufficient
capacity while avoiding needless complication. Faults must also be located
promptly, for a defective contact in a central office may interfere with many
subscribers at once.

The telephone exchange illustrates a general principle of modern engineering:
when the number of users becomes large, organization may be as important as
the individual mechanism. The usefulness of each instrument depends upon the
ability of the entire system to work together.""",

"ai07_valid_photography.txt": """THE DEVELOPMENT OF PHOTOGRAPHIC PLATES.

The photographic plate is often regarded merely as a surface upon which the
image of an object is received, but the preparation of that surface determines
many of the useful qualities of the photograph. The sensitive material must
respond sufficiently to light, remain attached to the glass, and preserve the
image during the processes required to develop and fix it.

A plate intended for one purpose may not be equally suitable for another. A
portrait photographer desires a pleasing rendering of delicate tones, while an
astronomer is concerned with the detection of faint points of light. In each
case the value of the plate depends upon the relation between its sensitivity
and the character of the work.

The chemical preparation is therefore not a minor matter. Small changes in the
composition of the coating may alter the speed of exposure or the manner in
which detail is recorded. Storage must likewise be considered, since the
sensitive surface may deteriorate if improperly kept.

The progress of photography will probably come not from a single universal
plate but from the production of materials adapted to definite uses. The
important improvement is greater command over the conditions under which the
image is formed and preserved.""",

"ai08_valid_weather.txt": """THE RECORDING OF WEATHER OBSERVATIONS.

The usefulness of meteorological observations depends greatly upon their
regularity. A thermometer read at one moment tells us something of the
temperature at that instant, but a succession of observations permits the
student to determine changes and recurring conditions. The same principle
applies to the pressure of the atmosphere, the direction of the wind and the
amount of cloud.

An observation is of little value if the position and condition of the
instrument are unknown. A thermometer exposed to the direct rays of the sun
may give a reading which cannot fairly be compared with one obtained in shade.
The instrument must therefore be placed according to a definite rule, and the
observer must follow the same method from day to day.

Long records are especially useful because individual disturbances then assume
their proper proportion. A single storm may attract attention because of its
violence, while a series of observations reveals whether such an event is rare
or forms part of a familiar seasonal pattern.

The object of systematic observation is not merely to accumulate figures. It
is to create a record from which changes may be distinguished from ordinary
variation. The value of a scientific observation lies as much in its manner of
collection as in the number which it finally produces.""",

"ai09_valid_mining_ventilation.txt": """VENTILATION IN DEEP MINES.

As mines are carried to greater depths, the circulation of air becomes a matter
of increasing importance. The workmen require a continuous supply of air fit
for breathing, while gases produced within the workings must be removed. A
shaft which appears ample when the mine is shallow may prove inadequate after
new galleries have been opened at greater depth.

The movement of air is influenced by differences of temperature and pressure,
but mechanical assistance is often necessary where natural circulation is
insufficient. The ventilating fan then becomes one of the principal machines
connected with the mine.

The arrangement of the passages presents a further difficulty. Air will seek
the path offering the least resistance, so that a poorly planned system may
send too much of the supply through one portion while leaving another
insufficiently ventilated. Doors, partitions and properly directed openings are
therefore essential.

A satisfactory ventilation system must be judged by the condition actually
maintained in the workings. The size of the fan alone does not establish that
the arrangement is effective. The quantity of air delivered, its distribution
and the condition of the return passages must all be considered together.""",

"ai10_test_optical_instruments.txt": """THE ACCURACY OF OPTICAL INSTRUMENTS.

An optical instrument may possess excellent lenses and still fail to give a
satisfactory result if the parts are not properly aligned. The path of the
light must be maintained with considerable precision, for a small deviation in
one component may produce a visible error in the final image.

Telescopes, microscopes and surveying instruments differ greatly in their
construction, yet the same general principle applies to each. The observer
relies upon a definite relation between the separate optical surfaces. Changes
of temperature may alter dimensions sufficiently to disturb this relation,
while rough handling may produce a displacement which is not immediately
obvious.

It is therefore desirable that instruments be examined under conditions
resembling those in which they are to be used. An adjustment which appears
perfect in the workshop may require correction when the instrument is exposed
to a different temperature or mounted upon another support.

Accuracy is thus not merely a property of the glass. It is a property of the
complete instrument, including its frame, adjustments and method of use. The
best construction is one which preserves the intended optical relations while
remaining sufficiently robust for practical work.""",

"ai11_test_agricultural_machinery.txt": """THE MODERN AGRICULTURAL MACHINE.

The introduction of machinery into agriculture has changed many operations
which formerly depended almost entirely upon manual labor. A machine may cut,
thresh or separate material with a regularity difficult to obtain by hand, but
its usefulness depends upon the circumstances of the farm as much as upon the
machine itself.

The soil, the crop, the available labor and the condition of the fields all
affect the result. A device which performs admirably under one set of
conditions may prove troublesome where the ground is uneven or the crop differs
in character. It is therefore dangerous to judge a machine solely by a
demonstration made under favorable circumstances.

Durability is another consideration. Agricultural machinery is exposed to
dust, moisture and irregular maintenance, and a complicated mechanism may lose
its advantage if delicate parts require frequent adjustment. Simplicity has a
value which cannot always be represented by a catalogue of improvements.

The most useful agricultural machine is consequently not necessarily the one
which performs the greatest number of operations. It is the one which performs
its intended work with sufficient speed, moderate expense and a reliability
suited to the conditions under which it must serve.""",

"ai12_test_urban_water_supply.txt": """THE PURIFICATION OF CITY WATER.

The growth of towns has increased the necessity for a dependable supply of
water suitable for household use. A source which appears perfectly clear is
not necessarily free from undesirable impurities, and the character of the
water may change after heavy rain or other disturbance.

The first requirement is therefore knowledge of the source. Rivers, wells and
reservoirs each possess different conditions, and the manner in which water
reaches the inhabitants may introduce further opportunities for contamination.
Storage tanks and pipes must be maintained as carefully as the original
supply.

Filtration provides one means of improving the quality of water. Its action
depends upon the passage of the liquid through a material capable of retaining
unwanted substances, but the operation must be watched if the filter is to
remain effective. A filter which is neglected may give a false appearance of
security.

The problem of water supply illustrates the necessity of considering an entire
system rather than one operation. The source, treatment, storage and
distribution are connected, and failure in any part may influence the quality
received at the end of the service. Sound public arrangements therefore depend
upon continued observation and maintenance rather than upon one isolated
improvement."""
}

out = Path("dataset/ai")

for name, text in articles.items():
    (out / name).write_text(text.strip() + "\n", encoding="utf-8")

print("Created", len(articles), "AI articles")

for path in sorted(out.glob("*.txt")):
    words = len(re.findall(r"\b\w+(?:['’]\w+)?\b",
                           path.read_text(encoding="utf-8")))
    print(path.name, words, "words")