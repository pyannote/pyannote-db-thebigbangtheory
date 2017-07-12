#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


import os.path as op
from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pandas import read_table
from pyannote.core import Annotation, Segment


class Season1(SpeakerDiarizationProtocol):

    def load_forced_alignment(self, uri):

        # load raw forced_alignment file
        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        path = op.join(data_dir, 'forced_alignment',
                       '{uri}.out'.format(uri=uri))
        names = ['start', 'duration', 'speaker', 'word', 'confidence']
        data = read_table(path, delim_whitespace=True, names=names)

        # create pyannote.core.Annotation
        annotation = Annotation(uri=uri, modality='speaker')
        for row in data.itertuples():
            segment = Segment(row.start, row.start + row.duration)
            annotation[segment] = row.speaker

        # merge adjacent segments from same speaker
        return annotation.support()


    def subset_iter(self, subset):

        data_dir = op.join(op.dirname(op.realpath(__file__)), 'data')
        path = op.join(data_dir, 'Season1.{subset}.lst'.format(subset=subset))
        with open(path, mode='r') as fp:
            uris = [line.strip() for line in fp.readlines()]

        for uri in sorted(uris):
            annotation = self.load_forced_alignment(uri)
            yield {
                'database': 'TheBigBangTheory',
                'uri': uri,
                'annotation': annotation
            }

    def trn_iter(self):
        return self.subset_iter('trn')

    def dev_iter(self):
        return self.subset_iter('dev')

    def tst_iter(self):
        return self.subset_iter('tst')


class TheBigBangTheory(Database):
    """TheBigBangTheory database"""

    def __init__(self, preprocessors={}, **kwargs):
        super(TheBigBangTheory, self).__init__(preprocessors=preprocessors, **kwargs)

        self.register_protocol(
            'SpeakerDiarization', 'Season1', Season1)
